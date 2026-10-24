# FrameFit 1.0.0 — Google Play release guide

This guide covers the prepared release candidate. It does not confirm Google Play approval or completed production publication.

| Release information | Value |
|---|---|
| App name | FrameFit |
| Application ID | `sk.framefit.app` |
| Version | 1.0.0 |
| Version code | 100 |
| Publisher | Gabriel Polak |
| Public support email | gabriel.polak.1986@gmail.com |
| Intended age groups | 13–15, 16–17, 18+ |
| Android support | Android 8.0+; target API 36 |

## 1. Prepare the account

- Complete identity, ownership and contact verification required by Play Console.
- Confirm production access and any account-specific testing requirement. Having a developer account does not by itself establish production eligibility.
- Select distribution countries and decide whether the app will be free or paid. The current build has no ads, in-app purchases or subscriptions; pricing is the publisher's decision.

## 2. Upload to internal testing

1. Create the FrameFit app and select the **Photography** category.
2. Set up **Play App Signing**. The prepared AAB has an upload signature; Google Play signs delivered installations with the app signing key.
3. If an upload certificate is already registered for this app, verify that it matches `signing/upload-certificate.pem`. If not, follow the applicable upload-key reset process before uploading.
4. Create an **Internal testing** release and upload `FrameFit-1.0.0.aab`.
5. Add release notes from `store-listings/<language>/release_notes.txt`.
6. Confirm the displayed version name and code. Upload `mapping.txt` if Console needs it for readable crash reports; retain the mapping for this exact build.
7. Configure testers and share the opt-in link with the people you intend to test the app.

The standalone `FrameFit-1.0.0-universal-test.apk` is for local testing. Upload the AAB, not a debug APK, to the Play release track.

## 3. Complete the store listing

| Prepared asset | Location in the release bundle |
|---|---|
| Title and descriptions | `store-listings/en/`, `store-listings/sk/`, `store-listings/cs/` |
| Release notes | `release_notes.txt` in each language directory |
| 512 × 512 icon | `store-assets/icon-512.png` |
| 1024 × 500 feature graphics | `store-assets/feature-en-1024x500.png` and Slovak/Czech variants |
| Four 1080 × 1920 screenshots per language | `store-assets/screenshots/en/`, `sk/`, `cs/` |
| Screenshot descriptions and provenance | `store-assets/screenshot-manifest.json` |

Use matching language assets for each listing. Screenshots show the real release UI on an emulator with original geometric demo artwork. Set the public support email to **gabriel.polak.1986@gmail.com**.

These paths refer to the prepared release bundle. Publishing this guide to GitHub does not automatically publish those assets or the app binaries.

## 4. Publish the privacy policy

The policy is prepared in [pull request #1](https://github.com/NyX3Ton/nyx3ton-project/pull/1).

1. Review and merge the policy pull request.
2. Open the repository's **Settings → Pages**.
3. Select **Deploy from a branch**, branch **main**, folder **/docs**, then save.
4. Wait for deployment and open the [planned public policy page](https://nyx3ton.github.io/nyx3ton-project/framefit/privacy/) without signing in.
5. Check that it loads successfully, has the correct publisher/contact details and offers English, Slovak and Czech versions.
6. Enter the working public URL in Play Console.

The planned URL had not been verified as live when this guide was prepared. A repository or pull request URL is not a substitute for the public policy page.

## 5. Complete App content declarations

These are proposed answers for version 1.0.0. The publisher must review the actual Console questions and confirm that the answers remain accurate.

### Ads and app access

- **Ads:** no.
- **App access:** all features available without an account or restricted login; reviewer credentials are not required.

### Target audience and rating

- Intended age groups: **13–15, 16–17 and 18+**. Do not select children under 13 for this release.
- Complete the IARC questionnaire according to the app's actual content. Intended audience age does not automatically determine the resulting content rating.

### Data safety

Proposed declarations: **no data collected** and **no data shared**, using Google's definitions and applicable exclusions.

Basis for these proposed answers:

- Photos, frame images, pattern samples and project settings are processed locally.
- The production build has no internet permission, accounts, ads, analytics or remote crash-reporting SDK.
- The photo picker exposes only images selected by the user.
- Sharing and printing are user initiated, with the receiving app/service chosen by the user. Assess these against Google's current user-initiated transfer rules.
- Optional support emails are handled outside the app as described in the privacy policy.

Recheck these answers whenever adding an SDK, network feature, account, telemetry or other data flow. The dependency inventory is in `legal/` in the release bundle.

### Permissions and accounts

- Camera permission supports user-selected photo, frame and pattern capture.
- The app does not request bulk photo-library access.
- This version has no account creation or login; account deletion does not apply to an account the app does not create.

The app is not intended as a news, health, financial, VPN or government app. Complete additional declarations according to the actual release and the publisher's circumstances.

## 6. Finish testing before production

- Review the **Pre-launch report** and fix blocking findings.
- If the account is subject to testing requirements for newer personal developer accounts, complete the closed test and apply for production access. The published requirement for personal accounts created after 13 November 2023 is at least **12 testers opted in continuously for 14 days**; verify applicability and current instructions in Console.
- Test on the intended **Samsung S26+** and representative additional devices.
- Check camera capture, denied permission, project restoration, large image exports and larger HEIC files where supported.
- Test sharing with the installed social apps that users will actually use.
- Check accessibility, large text, small displays and TalkBack.
- Print the 100 mm calibration square on a physical printer, measure it and check a final crop against a real frame.

Emulator results do not establish physical print accuracy or compatibility with a specific phone. Prepared verification records are in `verification/` in the release bundle.

## 7. Describe capabilities accurately

Version 1.0 uses the **CPU**. The NPU backend is experimental and development-only; vendor runtime libraries and verified hardware acceleration are not included in production.

Do not promise NPU acceleration, precise automatic physical measurement or arbitrary frame tracing. Recognition is an editable experimental suggestion. HEIC, sharing and printing depend on device capabilities and installed apps/services.

## 8. Protect signing material

The private upload key and passwords are stored locally in the protected `.release-secrets/` directory, excluded from the public release bundle and repository.

- Make an encrypted backup to a secure location you control, following the local `BACKUP-INSTRUCTIONS.txt`.
- Never publish keys/passwords, attach them to a release or paste them into chat.
- Retain the upload key for later updates; do not regenerate it for each release.
- A development build, a local upload-signed APK and a Play installation can have different signing certificates. Export needed results before uninstalling an incompatible earlier build; uninstalling deletes its private project.

## Official references

- [Target API requirements](https://developer.android.com/google/play/requirements/target-sdk)
- [Data safety form](https://support.google.com/googleplay/android-developer/answer/10787469?hl=en)
- [Testing requirements for new personal accounts](https://support.google.com/googleplay/android-developer/answer/14151465?hl=en)
- [Store graphics and screenshots](https://support.google.com/googleplay/android-developer/answer/9866151?hl=en)
- [Target audience and content](https://support.google.com/googleplay/android-developer/answer/9867159)

**Handover status:** prepared for internal testing. Public policy deployment, Console declarations, applicable testing, physical-device checks and Google's review remain release gates.
