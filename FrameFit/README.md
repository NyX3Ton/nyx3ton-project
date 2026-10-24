# FrameFit

**Your photo. Your frame.**

FrameFit is an Android app for preparing photos for circular, oval, rectangular and arched frames. Choose a photo, adjust the crop, then save, share or print it. Add seasonal decorations, create a border from your own pattern, or preview your photo inside a photographed frame.

**Photo → Frame → Finish**

| App information | |
|---|---|
| Prepared version | 1.0.0 · versionCode 100 |
| Requirements | Android 8.0 or later |
| Languages | English, Slovak and Czech |
| Appearance | System, light or dark |
| Photo processing | On your device; production version uses the CPU |
| Publisher | Gabriel Polak |
| Intended audience | Ages 13 and up |

> **Release status:** version 1.0.0 is prepared for internal testing. When this guide was prepared, it had not been approved or publicly released on Google Play. A public installation link will be added after publication.

## Contents

- [Features](#features)
- [Installation](#installation)
- [Your first photo](#your-first-photo)
- [Decorations and custom patterns](#decorations-and-custom-patterns)
- [Preview inside a real frame](#preview-inside-a-real-frame)
- [Printing and calibration](#printing-and-calibration)
- [Saving and sharing](#saving-and-sharing)
- [Language and appearance](#language-and-appearance)
- [Privacy and your project](#privacy-and-your-project)
- [Troubleshooting](#troubleshooting)
- [Development and release](#development-and-release)
- [Contact](#contact)

## Features

- Crop for a **circle, oval, rectangle or arch with a semicircular top**.
- Choose a photo or take one with a camera overlay.
- Move, zoom and rotate the photo within its opening.
- Choose from **52 decorations** with three strengths: subtle, classic and bold.
- Create a repeating border from a photo of fabric, a drawing, paper or another pattern.
- Preview a photo inside a photographed frame with manual perspective adjustment.
- Export JPG/PNG, use HEIC where supported, create PDFs, share and print.
- Automatically save one work-in-progress project on your device.

## Installation

For internal testing, use the Google Play invitation supplied by the publisher. If testing a standalone APK, use the file provided by the publisher. An **AAB cannot be installed directly on a phone**; it is uploaded to Google Play.

The app opens with a three-step guide. Selecting an existing photo uses Android's photo picker. Camera permission is needed when taking a photo, frame image or pattern image; it is not needed to select an existing picture.

**Upgrading from a development build:** Android may reject an update signed with a different key. Export any results you want to keep before uninstalling the previous app. Uninstalling deletes its private work-in-progress project. A standalone test APK and a Google Play installation may have different signing certificates.

## Your first photo

### 1. Photo

1. Tap **Choose a photo** or **Take a photo**.
2. Select an image or capture a new one.
3. Drag with one finger to move it. Pinch with two fingers to zoom.
4. Tap **Choose a frame**.

Your original photo in the gallery is not modified.

### 2. Frame

1. Select the shape you want.
2. Check that important parts of the photo remain inside the crop.
3. Optionally tap **Add a little decoration**.
4. Tap **Looks good to me**.

A circle has equal width and height. An oval uses different dimensions; switching from a circle restores an oval aspect ratio.

### 3. Finish

- **Save my photo** saves a JPG to your gallery.
- **Send to someone** opens Android's sharing options.
- **Print** asks you to confirm dimensions before opening Android's print dialog.

Scroll the page if a secondary option is outside the visible area. Tap **Back** to return to editing.

## Decorations and custom patterns

### Built-in decorations

In the Frame step, tap **Add a little decoration**, then **All decorations and your own patterns** for the full collection. Search for a theme and adjust the decoration strength.

Themes include Christmas, Halloween, Easter, St Patrick's Day, Lunar New Year, Diwali, Holi, Hanukkah, Eid, Thanksgiving, birthdays and weddings. You can also leave the photo undecorated.

### A border made from your own photo

1. Open **All decorations and your own patterns**, then the custom pattern section.
2. Photograph a pattern or choose a pattern image from your phone.
3. Move and zoom the image to select a square sample. Rotate it if needed.
4. Confirm the sample to apply it as a border.
5. Open **Adjust custom pattern** to change the border width, repetition size and tile mirroring.

Try wrapping paper, embroidery, wood, a drawing or a favourite fabric. Even lighting and a clean detail make the pattern easier to repeat. Mirroring can reduce visible seams.

The sample repeats together with its background; the app does not automatically remove that background. Choosing a built-in decoration disables the custom pattern. You can enable the saved pattern again later.

## Preview inside a real frame

Open **Options → Open full studio**. In the frame section, use the frame photography/recognition action or import a frame image.

1. Photograph the whole frame as straight-on as possible, avoiding reflections from glass.
2. Place a contrasting background behind an empty opening so its inner edge is clearer.
3. Inspect the suggested opening. Use the alignment action when you want to apply it.
4. Correct the shape and alignment manually if necessary. Four draggable handles adjust the perspective of the preview.
5. Check the inserted photo and save the frame preview if you want to keep it.

**Recognition is an experimental edge-based suggestion.** A photograph does not establish physical dimensions. Measure the opening with a ruler before printing. The app supports four geometric shapes; it does not trace arbitrary irregular openings.

A photographed-frame preview is a visualisation. To print a photo that will go inside the physical frame, select the cropped photo itself as the output.

## Printing and calibration

### Simple printing

1. Measure the frame's **visible inner opening**, not its outside dimensions.
2. In the Finish step, tap **Print**.
3. Enter width and height in centimetres, or the diameter for a circle.
4. Choose A4 or 10 × 15 cm paper and continue to Android's print dialog.
5. Check the paper size in the print service too. If scaling controls are available, choose **100% / actual size**.

Example: an opening 8 cm wide and 12 cm high needs **8 × 12 cm**, regardless of the outside size of the frame. Some full-studio measurements use millimetres: 8 cm = 80 mm.

### Test outline and calibration

Before printing the final photo, print a test outline and check it against the frame. Advanced printing options also include a **100 mm calibration** square.

1. Print the calibration square.
2. Measure one side with a ruler.
3. If it is not exactly 100 mm, enter the measured length and apply the scale correction.
4. Check the test outline again, then print the final photo.

Correction applies to the current project and print setup. Check it again after changing the printer, paper or print service. The full studio also lets you set an overlap beneath the frame.

The printer must be available through a compatible Android print service. FrameFit does not include individual USB/Bluetooth printer drivers. Alternatively, save a PDF and print from another device. The print service can affect physical scale independently of the app.

## Saving and sharing

**Save my photo** creates a JPG. Open **More ways to save** for additional formats and settings.

| Output | Typical use |
|---|---|
| JPG | Gallery photos and everyday sharing |
| PNG | Lossless image output; files may be larger |
| HEIC | Space-efficient output, where the device can encode it |
| PDF | Printing with specified dimensions |
| Frame preview | Showing the photo inside your photographed frame |

Choose the destination in Android's share sheet. Instagram, Facebook, TikTok and other apps appear only if installed and able to accept the image type. FrameFit does not publish posts automatically. If an app is missing, save a JPG and select it from that app instead.

## Language and appearance

Open **Options → Language and appearance**.

- Follow the system language or choose English, Slovak or Czech manually.
- Follow the system appearance or select a light or dark theme.
- Use system colours on Android 12 or later.
- The app follows the system font size. Scroll longer pages to reach more controls.

## Privacy and your project

FrameFit processes photos locally. Version 1.0 has no accounts, ads, analytics or internet permission. The photo picker exposes only images you select. Sharing and printing use destinations you choose; those services may have their own network access and terms.

One work-in-progress project is saved in private app storage. It can contain the original photo, working preview, frame image, selected pattern sample and editing settings. **This is one project, not a gallery backup or a library of several projects.**

Delete the project through the advanced options and confirm deletion. This does not delete images already exported to the gallery or copies sent to other apps. Uninstalling removes the app's private data. Export results you want to keep first.

The full privacy policy and third-party notices are available in the app's settings. The public policy is prepared in [pull request #1](https://github.com/NyX3Ton/nyx3ton-project/pull/1). Its [planned website address](https://nyx3ton.github.io/nyx3ton-project/framefit/privacy/) requires the change to be merged and GitHub Pages enabled; the address had not been verified as live when this guide was prepared.

## Troubleshooting

| Problem | What to try |
|---|---|
| The detected opening does not fit | Use a contrasting background, even light and a straight-on view. Avoid reflections. Correct the suggestion manually. |
| The print is too large or small | Check cm/mm, inner dimensions, paper size and print-service scaling. Disable fit-to-page if available and measure the 100 mm calibration square. |
| Circle and oval look similar | A circle needs equal width and height; an oval needs different dimensions. Check the preview and measurements as well as the small icon. |
| HEIC export fails | Use JPG or PNG. HEIC depends on the device's encoder; large exports need testing on the actual phone. |
| The printer is missing | Check the Android print service and printer availability, or save a PDF and print it through another app or computer. |
| A social app is missing | Save a JPG and select it directly from the destination app. |

### Does FrameFit use my phone's NPU?

**Production version 1.0 uses the CPU.** The experimental development path considers MediaTek, Snapdragon, Exynos and Google Tensor, but hardware NPU acceleration is not a completed release feature. Having an NPU in the phone does not enable it automatically.

### Can I keep several unfinished projects?

This version keeps one work-in-progress project. Save finished results separately to the gallery or files.

## Development and release

### Source project

FrameFit uses Kotlin, Jetpack Compose, CameraX and Android's system interfaces. It requires JDK 17+, Android SDK 36 and Build Tools 36.0.0.

Run these commands **from the FrameFit source project root**, where `gradlew.bat` is located. This guide's presence in the shared repository does not mean the Android sources or binaries have already been published there.

```powershell
.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
```

The development APK is written to `app/build/outputs/apk/debug/app-debug.apk`. It is for development, not Google Play publication.

If the project's local tools have already been prepared, create a signed production build with:

```powershell
.\tools\build_release.ps1
```

This builds the production AAB/APK, runs release unit tests and Android Lint, and verifies the AAB signature. It requires an upload key. Keep private keys, passwords, `.release-secrets/` and local settings out of the public repository. Back up the key securely; do not regenerate it for each release.

### Recorded verification

Version 1.0.0, recorded on **24 October 2026**:

- 29 passing release unit tests.
- 22 passing debug-variant Android tests on an Android 16 / API 36 emulator.
- Release Android Lint: 0 errors and 28 warnings.
- Signed AAB validated with bundletool; an optimised APK generated from it and installed.
- Release UI checks for photo selection, a circular crop, JPG saving, Android print preview, PDF saving and opening third-party notices.
- Static 16 KB alignment checks for 64-bit libraries. This is not a runtime test on a 16 KB device.

A physical S26+, camera and printer were not part of this verification. Before production, finish practical checks for camera behaviour, large HEIC exports, social apps, accessibility and physical print scale.

### Google Play

The app ID is `sk.framefit.app`. Upload the signed **AAB** to internal testing. Complete the declarations, publish the privacy policy, meet account testing requirements and obtain Google's review before public release.

See [PLAY-CONSOLE.md](PLAY-CONSOLE.md) for the handover checklist and remaining release gates. Use the current instructions in Play Console when submitting. A signed package alone is not Google Play approval.

## Contact

**Gabriel Polak** · [gabriel.polak.1986@gmail.com](mailto:gabriel.polak.1986@gmail.com)

When reporting a problem, include your phone model, Android version, FrameFit version and reproduction steps. For printing issues, add the printer model, paper size and print service. Do not send passwords or private photos unless a photo is needed to explain the problem and you choose to provide it.
