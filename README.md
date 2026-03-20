```
 ██╗   ██╗ ██████╗ ███████╗██████╗ ███████╗██████╗
 ██║   ██║╚════██╗██╔════╝██╔══██╗██╔════╝██╔══██╗
 ██║   ██║ █████╔╝███████╗██████╔╝█████╗  ██████╔╝
 ╚██╗ ██╔╝ ╚═══██╗╚════██║██╔═══╝ ██╔══╝  ██╔══██╗
  ╚████╔╝ ██████╔╝███████║██║     ███████╗██║  ██║
   ╚═══╝  ╚═════╝ ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═╝
```

# V3SP3R — The AI Brain for Your Flipper Zero

> **Talk to your Flipper Zero like it's a teammate.** Vesper turns your pocket hacking tool into an AI-powered command center — controlled entirely through natural language from your Android device.

No menus. No manuals. Just tell it what you want.

---

## Why Vesper?

The Flipper Zero is one of the most versatile hardware hacking tools ever made — but navigating its menus, managing files, and crafting signals by hand is tedious. **Vesper eliminates the friction.** Plug in an AI brain via OpenRouter, connect over Bluetooth, and you have a voice-commanded hardware lab in your pocket.

- **Instant expertise** — Don't memorize SubGHz protocols or IR formats. Just say what you want.
- **Real-time control** — The AI reads your Flipper's state, executes commands, and reports back in seconds.
- **Signal alchemy** — Build, layer, and export custom RF waveforms with a visual editor.
- **Ops-grade reliability** — Built-in diagnostics, runbooks, and macro recording for repeatable workflows.
- **Safety-first architecture** — Every AI action is risk-classified. Destructive operations require explicit confirmation. System paths are locked by default.

Whether you're a security researcher, a red teamer, a CTF competitor, a hardware tinkerer, or just someone who wants to understand the invisible signals around you — Vesper makes the Flipper Zero *dramatically* more accessible and more powerful.

---

## Recommended AI Models

Vesper works with any model on [OpenRouter](https://openrouter.ai). For the best experience, use the latest frontier models:

| Model | Why Use It | Speed | Cost |
|-------|-----------|-------|------|
| **`nousresearch/hermes-4`** | Outstanding tool-use and instruction following — purpose-built for agentic workflows like Vesper. Top pick for power users. | Fast | $$ |
| **`anthropic/claude-opus-4`** | The most capable reasoning model available. Exceptional at complex multi-step Flipper operations and signal analysis. | Medium | $$$$ |
| **`anthropic/claude-sonnet-4`** | Best balance of speed, intelligence, and cost. Great default choice. | Fast | $$ |
| **`anthropic/claude-haiku-4`** | Blazing fast for simple reads and quick commands. Cheapest option. | Fastest | $ |
| **`openai/gpt-4o`** | Strong general-purpose alternative. | Fast | $$ |

**Our recommendation:** Start with **Hermes 4** or **Claude Sonnet 4** for daily use. Reach for **Claude Opus 4** when you need deep reasoning — complex signal crafting, multi-step file operations, or creative BadUSB scripting.

---

## What Can You Do With It?

### Chat — AI-Driven Flipper Control
Talk to your Flipper in plain English:
- *"Show me my SubGHz captures"*
- *"What's my battery level?"*
- *"Create a backup of all my IR remotes"*
- *"Generate a BadUSB script that opens a reverse shell"*
- *"Change the frequency in garage.sub to 315MHz"*

### Ops Center — Operational Command Surface
Built for reliability-obsessed users:
- **Pipeline Health** — BLE/RPC/CLI readiness and diagnostics at a glance
- **Runbooks** — One-tap recovery and smoke-test sequences
- **Macro Recorder** — Record and replay button workflows with precise timing
- **Live Status** — Transport and command pipeline behavior in one view

### Alchemy Lab — Signal Synthesis
Build custom RF signals from scratch:
- Visual waveform editor with real-time preview
- Layer and fuse multiple signal patterns
- Export directly to your Flipper's SD card

### Device Manager
Full Flipper visibility:
- Battery, storage, and connection status
- Scan, pair, and manage BLE connections
- Direct file browsing and management

### Risk & Permissions Engine
Every AI action is classified before execution:
- **Low risk** — Read-only ops execute automatically
- **Medium risk** — File writes show a diff for review
- **High risk** — Destructive ops (delete, move, overwrite) require double-tap confirmation
- **Blocked** — System/firmware paths require explicit unlock

Configure **auto-approve** per risk tier in Settings to move faster when you trust the workflow.

---

## Quick Start

### What You Need

| Item | Notes |
|------|-------|
| **Flipper Zero** | [shop.flipperzero.one](https://shop.flipperzero.one) |
| **Android Phone** | Android 8.0+, Bluetooth required |
| **Computer** | Windows, Mac, or Linux — to build the app |
| **OpenRouter Account** | Free signup, pay-per-use — [openrouter.ai](https://openrouter.ai) |

### 1. Prep Your Flipper
1. Charge it up (USB-C)
2. Update firmware via [qFlipper](https://flipperzero.one/update) (recommended)
3. Enable Bluetooth: Settings → Bluetooth → ON

### 2. Get an OpenRouter API Key
1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Go to **Keys** → **Create Key**
3. Copy the key (`sk-or-...`) — you'll paste this into Vesper later
4. Add $5–10 in credits to start (most conversations cost pennies)

### 3. Build the App

```bash
git clone https://github.com/elder-plinius/V3SP3R.git
cd V3SP3R
```

Open the project in [Android Studio](https://developer.android.com/studio), let Gradle sync, then:
- **Build → Build APK(s)** or hit the green hammer
- APK output: `app/build/outputs/apk/debug/app-debug.apk`

<details>
<summary><strong>Don't have Git?</strong></summary>

- **Windows**: [git-scm.com](https://git-scm.com) → install → use Git Bash
- **Mac**: Type `git` in Terminal — it'll prompt you to install
- **Linux**: `sudo apt install git`
</details>

### 4. Install on Your Phone

**Via USB (recommended):**
1. Enable Developer Options: Settings → About Phone → tap "Build Number" 7 times
2. Enable USB Debugging in Developer Options
3. Connect phone via USB → hit Play (▶) in Android Studio

**Via APK transfer:**
1. Copy `app-debug.apk` to your phone
2. Open it → allow "Install unknown apps" if prompted → Install

### 5. First Launch
1. **Grant permissions** — Bluetooth, Location (required for BLE scanning), Notifications
2. **Add your API key** — Device tab → Settings → paste your OpenRouter key
3. **Connect** — Device tab → Scan → tap your Flipper
4. **Go** — Chat tab → start talking to your Flipper

---

## Example Commands

**Basics:**
```
"What's on my SD card?"
"How much storage do I have left?"
"What's my battery at?"
```

**File operations:**
```
"Show me my SubGHz folder"
"Read my garage.sub file"
"Create a new folder called backups"
"Copy all .sub files to backups"
```

**Advanced:**
```
"Change the frequency in garage.sub to 315MHz"
"Generate a BadUSB script that opens notepad and types hello"
"List all IR remotes and back them up"
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Vesper App                  │
├─────────────────────────────────────────┤
│  UI Layer (Jetpack Compose)             │
│  ├── Chat Screen                        │
│  ├── Ops Center                         │
│  ├── Alchemy Lab                        │
│  ├── File Browser                       │
│  └── Device Screen                      │
├─────────────────────────────────────────┤
│  Domain Layer                           │
│  ├── VesperAgent (AI Orchestration)     │
│  ├── CommandExecutor (Risk Enforcement) │
│  ├── Signal Processing                  │
│  └── Device Automation + Auditing       │
├─────────────────────────────────────────┤
│  Data Layer                             │
│  ├── OpenRouter Client                  │
│  ├── Flipper BLE Service                │
│  └── Encrypted Storage                  │
└─────────────────────────────────────────┘
```

## Project Structure

```
app/src/main/java/com/vesper/flipper/
├── ai/                    # AI integration
│   └── OpenRouterClient.kt
├── ble/                   # Bluetooth
│   ├── FlipperBleService.kt
│   └── FlipperFileSystem.kt
├── domain/model/          # Data models
│   ├── Alchemy.kt         # Signal synthesis
│   ├── Chimera.kt         # Signal fusion
│   └── Signal.kt          # Capture formats
├── security/              # Security utilities
│   └── SecurityUtils.kt
├── ui/
│   ├── screen/            # UI screens
│   │   ├── ChatScreen.kt
│   │   ├── OpsCenterScreen.kt
│   │   └── AlchemyLabScreen.kt
│   └── viewmodel/         # State management
└── MainActivity.kt
```

---

## Troubleshooting

<details>
<summary><strong>Flipper not found when scanning</strong></summary>

1. On Flipper: Settings → Bluetooth → make sure it's ON
2. Toggle Bluetooth off/on on your phone
3. Make sure Flipper isn't connected to another device
4. Move within 3 feet
</details>

<details>
<summary><strong>Build failed in Android Studio</strong></summary>

1. File → Sync Project with Gradle Files
2. Build → Clean Project → Rebuild Project
3. Nuclear option: close Android Studio, delete `.gradle` folder, reopen
</details>

<details>
<summary><strong>App crashes on launch</strong></summary>

1. Ensure all permissions are granted
2. Uninstall and reinstall
3. Confirm Android 8.0+
</details>

<details>
<summary><strong>AI not responding</strong></summary>

1. Verify your OpenRouter API key
2. Check your OpenRouter credit balance
3. Check internet connection
</details>

<details>
<summary><strong>Permission denied errors</strong></summary>

- Some Flipper paths are protected by default (system files, firmware areas)
- Go to Settings → Permissions to unlock specific paths
- Enable auto-approve per risk tier to move faster
- Blocked paths always require manual unlock regardless of auto-approve settings
</details>

---

## Safety & Legal

- Vesper is a tool for **education and legitimate security research**
- Only use on devices you own or have explicit authorization to test
- All AI actions are logged and auditable
- The AI refuses clearly malicious requests
- Destructive operations require explicit user confirmation

---

## Contributing

PRs welcome. Areas that need love:
- iOS version
- Signal format parsers
- New attack presets & templates
- UI/UX improvements

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Disclaimer

Vesper is for educational purposes and authorized security research. Users are responsible for complying with all applicable laws. The authors assume no liability for misuse.

---

**V3SP3R** — AI-powered hardware hacking, in your pocket. Your Flipper Zero just got a brain upgrade.
