# Screenshot demo inputs

Images from **ContextRico-DP** (Zenodo record 15036779, licensed CC-BY-4.0; original filenames kept). They are
used only as evaluation / demo inputs - ClearCart was never trained on them. All come from the **TEST** split of
`ml/eval_screenshots.py`.

Predictions below were produced by that script with OCR config `v4_psm3` (grayscale + 2x upscale + block merging, but auto layout (psm 3)) and
flagging threshold p_dark >= 0.9. ClearCart reads text only; visual patterns in these images are not analysed.

## `www.gasbike.net.png` - handled correctly

- Ground-truth labels: Countdown Timer, Limited Time Message
- Text-observable ClearCart categories expected: urgency
- Split: test (Web)
- ClearCart flags (top by p_dark):
  - `Add to wishlist HURRY! SALE ENDS IN: Days Hours Minutes Seconds` -> **urgency**, p_dark=1.0 (block)
  - `HURRY! SALE ENDS IN:` -> **urgency**, p_dark=1.0 (line)
  - `Days Hours Minutes Seconds` -> **urgency**, p_dark=1.0 (line)
  - `Oo 00 80868` -> **urgency**, p_dark=0.957 (line)

## `www.aafnation.com.png` - handled correctly

- Ground-truth labels: Low Stock Message, Countdown Timer
- Text-observable ClearCart categories expected: scarcity, urgency
- Split: test (Web)
- ClearCart flags (top by p_dark):
  - `HURRY! ONLY & LEFT IN STOCK.` -> **scarcity**, p_dark=1.0 (line)
  - `GUARANTEE PRINTED BY PRINT BRAINS HURRY! ONLY & LEFT IN STOCK.` -> **scarcity**, p_dark=0.948 (block)

## `knockaround.com.png` - handled correctly

- Ground-truth labels: Activity Message, Low Stock Message
- Text-observable ClearCart categories expected: scarcity, social_proof
- Split: test (Web)
- ClearCart flags (top by p_dark):
  - `Limited stock! ¢) 1 person is viewing this, 2 recently purchased it, and 1 has this in their cart.` -> **social_proof**, p_dark=1.0 (block)
  - `Limited stock! ¢) 1 person is viewing this, 2 recently purchased it, and 1 has this in` -> **social_proof**, p_dark=1.0 (line)
  - `$28.00 atterpay available for orders over $35 ©` -> **urgency**, p_dark=0.948 (line)
  - `Earn 140 Knock Points for this purchase!` -> **social_proof**, p_dark=0.929 (line)

## `(177)www.adoreme.com_a719.jpg` - MISSED

- Ground-truth labels: Countdown Timer
- Text-observable ClearCart categories expected: urgency
- Split: test (Web)
- ClearCart flags: nothing (no OCR candidate reached the threshold)
