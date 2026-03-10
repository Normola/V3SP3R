package com.vesper.flipper.voice

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import com.vesper.flipper.data.SettingsStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.concurrent.TimeUnit
import javax.inject.Inject
import javax.inject.Singleton

/**
 * ElevenLabs text-to-speech service with streaming audio playback.
 * Sends text to the ElevenLabs API and plays the returned audio in real-time.
 */
@Singleton
class ElevenLabsTtsService @Inject constructor(
    private val settingsStore: SettingsStore
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(15, TimeUnit.SECONDS)
        .build()

    private val _state = MutableStateFlow<TtsState>(TtsState.Idle)
    val state: StateFlow<TtsState> = _state.asStateFlow()

    private var currentTrack: AudioTrack? = null
    private var currentJob: Job? = null

    /**
     * Speak the given text using ElevenLabs TTS.
     * Streams PCM audio for lowest-latency playback.
     */
    suspend fun speak(text: String) {
        // Don't speak empty text
        if (text.isBlank()) return

        // Stop any current playback
        stop()

        val apiKey = settingsStore.elevenLabsApiKey.first()
        if (apiKey.isNullOrBlank()) {
            _state.value = TtsState.Error("ElevenLabs API key not configured")
            return
        }

        val voiceId = settingsStore.ttsVoiceId.first()

        _state.value = TtsState.Loading

        coroutineScope {
            currentJob = launch(Dispatchers.IO) {
                try {
                    streamAndPlay(apiKey, voiceId, text)
                } catch (e: Exception) {
                    if (_state.value !is TtsState.Idle) {
                        _state.value = TtsState.Error(e.message ?: "TTS playback failed")
                    }
                }
            }
        }
    }

    /**
     * Stop current playback immediately.
     */
    fun stop() {
        currentJob?.cancel()
        currentJob = null
        try {
            currentTrack?.apply {
                pause()
                flush()
                release()
            }
        } catch (_: Exception) {}
        currentTrack = null
        _state.value = TtsState.Idle
    }

    /**
     * Check if TTS is available (API key configured).
     */
    suspend fun isAvailable(): Boolean {
        val enabled = settingsStore.ttsEnabled.first()
        val apiKey = settingsStore.elevenLabsApiKey.first()
        return enabled && !apiKey.isNullOrBlank()
    }

    private suspend fun streamAndPlay(apiKey: String, voiceId: String, text: String) {
        // Clean text for TTS: strip markdown artifacts, image descriptions, tool output
        val cleanedText = cleanTextForSpeech(text)
        if (cleanedText.isBlank()) {
            _state.value = TtsState.Idle
            return
        }

        // Request PCM 24kHz 16-bit mono for direct AudioTrack playback (no decoder needed)
        val requestJson = """
            {
                "text": ${escapeJson(cleanedText)},
                "model_id": "eleven_flash_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.3
                }
            }
        """.trimIndent()

        val request = Request.Builder()
            .url("$API_BASE/text-to-speech/$voiceId/stream?output_format=pcm_24000")
            .addHeader("xi-api-key", apiKey)
            .addHeader("Content-Type", "application/json")
            .post(requestJson.toRequestBody("application/json".toMediaType()))
            .build()

        val response = client.newCall(request).execute()

        if (!response.isSuccessful) {
            val body = response.body?.string() ?: "Unknown error"
            _state.value = TtsState.Error("ElevenLabs API error ${response.code}: $body")
            response.close()
            return
        }

        val inputStream = response.body?.byteStream() ?: run {
            _state.value = TtsState.Error("Empty response from ElevenLabs")
            response.close()
            return
        }

        // Set up AudioTrack for PCM 24kHz 16-bit mono playback
        val sampleRate = 24000
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        ).coerceAtLeast(4096)

        val audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufferSize * 2)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        currentTrack = audioTrack
        audioTrack.play()
        _state.value = TtsState.Speaking

        try {
            val buffer = ByteArray(bufferSize)
            var bytesRead: Int

            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                if (!kotlinx.coroutines.coroutineContext.isActive) break
                audioTrack.write(buffer, 0, bytesRead)
            }

            // Wait for remaining audio to finish playing
            // Write silence to flush the buffer
            withContext(Dispatchers.IO) {
                audioTrack.write(ByteArray(bufferSize), 0, bufferSize)
            }
        } finally {
            inputStream.close()
            response.close()
            try {
                audioTrack.stop()
                audioTrack.release()
            } catch (_: Exception) {}
            currentTrack = null
            if (_state.value is TtsState.Speaking) {
                _state.value = TtsState.Idle
            }
        }
    }

    /**
     * Clean text for speech: remove markdown, code blocks, image descriptions, etc.
     */
    private fun cleanTextForSpeech(text: String): String {
        return text
            // Remove code blocks
            .replace(Regex("```[\\s\\S]*?```"), "")
            // Remove inline code
            .replace(Regex("`[^`]+`"), "")
            // Remove image descriptions from vision preprocessing
            .replace(Regex("\\[Attached image:.*?]"), "")
            .replace(Regex("\\[\\d+ image\\(s\\).*?]"), "")
            // Remove markdown bold/italic
            .replace(Regex("\\*{1,3}([^*]+)\\*{1,3}"), "$1")
            // Remove markdown headers
            .replace(Regex("^#{1,6}\\s+", RegexOption.MULTILINE), "")
            // Remove markdown links [text](url)
            .replace(Regex("\\[([^]]+)]\\([^)]+\\)"), "$1")
            // Remove markdown bullet points
            .replace(Regex("^\\s*[-*+]\\s+", RegexOption.MULTILINE), "")
            // Collapse whitespace
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
    }

    private fun escapeJson(text: String): String {
        val escaped = text
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")
        return "\"$escaped\""
    }

    companion object {
        private const val API_BASE = "https://api.elevenlabs.io/v1"

        // Premade voice IDs
        const val VOICE_CHARLOTTE = "iP95p4xoKVk53GoZ742B" // Swedish, seductive
        const val VOICE_ALICE = "9BWtsMINqrJLrRacOk9x"     // British, confident
        const val VOICE_ARIA = "pqHfZKP75CvOlQylNhV4"       // American, expressive
        const val VOICE_SARAH = "EXAVITQu4vr4xnSDxMaL"      // American, soft
        const val VOICE_LILY = "pFZP5JQG7iQjIQuC4Bku"       // British, warm

        val AVAILABLE_VOICES = listOf(
            VoiceOption(VOICE_CHARLOTTE, "Charlotte", "Swedish, seductive"),
            VoiceOption(VOICE_ALICE, "Alice", "British, confident"),
            VoiceOption(VOICE_ARIA, "Aria", "American, expressive"),
            VoiceOption(VOICE_SARAH, "Sarah", "American, soft"),
            VoiceOption(VOICE_LILY, "Lily", "British, warm")
        )
    }
}

data class VoiceOption(
    val id: String,
    val name: String,
    val description: String
)

sealed class TtsState {
    data object Idle : TtsState()
    data object Loading : TtsState()
    data object Speaking : TtsState()
    data class Error(val message: String) : TtsState()
}
