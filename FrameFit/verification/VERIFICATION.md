# Overenie FrameFit 1.0.0 (100)

24. október 2026. Podpísaný kandidát pre Google Play Internal testing. Identifikátor sk.framefit.app, Android 8+ (min26), target36.

- Optimalizovaná release zostava: R8, zmenšené zdroje, vypnuté ladenie, samostatný upload podpis.
- 29 jednotkových testov release variantu: úspech.
- 22 Android testov debug variantu: úspech na Android16/API36 x86_64, posledný beh 55,177s. Zahŕňajú lokalizáciu, vzhľad, sprievodcu, geometriu, exporty, projekty, politiku a licencie. Nie sú to inštrumentačné testy minifikovaného release APK.
- Release lint: bez chýb; presné počty upozornení sú v distribučnom verification/summary.json a XML reporte.
- Finálny AAB validovaný oficiálnym bundletool; z AAB vytvorený a podpísaný univerzálny APK, nainštalovaný na izolovaný emulátor. Overený Photo Picker, úprava výrezu, uloženie JPG do galérie a systémový tlačový náhľad.
- Oprávnenia CAMERA a vlastné podpisové oprávnenie AndroidX; bez INTERNET a prístupu ku celej galérii.
- Produkcia používa iba CPU. LiteRT a experimentálny NPU backend sú výhradne v debug variante. Licenčný audit zahŕňa 87 release závislostí, bez chýbajúcich licenčných metadát; pôvodné oznámenia sú priamo v aplikácii.
- Šesť 64-bitových knižníc má ELF PT_LOAD zarovnanie najmenej16384 bajtov; zipalign -P16 úspešný. Ide o statickú kontrolu, nie spustenie na 16KB zariadení.
- Store grafika: pôvodná ikona512x512 a lokalizované grafiky1024x500, sk/cs/en popisy. Snímky sú zo skutočnej aplikácie na emulátore.

## Praktické limity

S26+, fyzický fotoaparát a tlačiareň neboli súčasťou tohto overenia. Treba skontrolovať veľké HEIC exporty, prijatie fotiek nainštalovanými sociálnymi aplikáciami, TalkBack a reálny100mm kalibračný výtlačok. Rozpoznanie otvoru je experimentálne a nenahrádza meranie.

Politika súkromia je pripravená v PR; verejná Pages URL ešte musí byť zverejnená a overená. Google Play schválenie, Pre-launch report a prípadný12testerov/14dní uzavretý test sú samostatné kroky. Podklady: release/PLAY-CONSOLE.md. Výsledky a kontrolné súčty konkrétneho odovzdávaného balíka: dist/release-1.0.0/.

História prototypu je vo VERIFICATION_0.6.md.
