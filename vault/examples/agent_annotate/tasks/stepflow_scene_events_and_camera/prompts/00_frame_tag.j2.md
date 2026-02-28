You are an expert AI for visual scene analysis. Analyze the image based on the rules below and provide a JSON output.

**INSTRUCTIONS:**
1.  First, evaluate the image for `content-assessment`.
2.  If `content-assessment` is `meaningful-content`, complete all subsequent categories.
3.  If `content-assessment` is `credits-or-titles` or `solid-color-screen`, set all other category values to `null` or `[]`.

**TAGGING SCHEMA:**

VLM Image Analysis Schema

`0. content-assessment`
- Rule: Select one.
- Tags: `meaningful-content`, `solid-color-screen`, `test-card` (color bars/noise), `credits-or-titles`, `split-screen`

`1. quality-flags`
- Rule: Select any, or none.
- Tags:
  - `out-of-focus` (Unintentional blur)
  - `motion-blur` (Speed effect)
  - `low-res-pixelated` (Blocky/low quality)
  - `compression-artifacts` (JPEG/Video encoding noise)
  - `visual-noise` (Grainy/High ISO)
  - `overexposed` (Too bright), `underexposed` (Too dark)
  - `watermark-or-logo`, `heavy-subtitles`, `ui-overlay`, `letterbox-pillarbox` (Black bars)

`2. spatial-domain`
- Rule: Select one. Primary location type.
- Tags: `indoor`, `outdoor`, `vehicle-interior`, `outer-space`, `underwater`, `stage-theatrical`, `studio-backdrop`, `liminal-space`, `surreal-abstract`, `digital-virtual`

`3. scene-atmosphere`
- Rule: Select any that apply.
- Tags: `day`, `night`, `dusk`, `dawn`, `golden-hour`, `overcast`, `natural-light`, `artificial-light`, `hard-light` (sharp shadows), `neon-lighting`, `silhouette`, `rain`, `snow`, `foggy`

`4. shot-scale`
- Rule: Select one. Follow priority: Single Subject > Group > Object/Scenery.
- Tags:
  1. 广阔视野 (Wide View)
  - `extreme-long-shot` (ELS)
    - [Human]: Person is a tiny dot; indistinguishable.
    - [Scenery]: Vast landscape, city skyline, planet view.
  - `long-shot` (LS / Wide Shot)
    - [Human]: Full body visible + plenty of headroom/ground; environment dominates.
    - [Group]: Shows the entire crowd/formation.
    - [Object]: Shows a whole building, a large car, or a whole room.

  2. 全貌展示 (Full View)
  - `full-shot` (FS)
    - [Human]: Head to toe fits roughly in the frame.
    - [Group]: 2-3 people visible from head to toe.
    - [Object]: An object fits fully (e.g., a chair, a bicycle, a door).
  - `medium-full-shot` (MFS / Cowboy Shot)
    - [Human]: Knees up.
    - [Group]: 2-3 people from knees up.
    - [Object]: Focus on a large section of an object (e.g., car front half).

  3. 交互距离 (Interaction View)
  - `medium-shot` (MS)
    - [Human]: Waist up.
    - [Group]: Interactions between 2 people (waist up).
    - [Object]: A tabletop setup, a computer monitor, a messy desk.
  - `medium-close-up` (MCU)
    - [Human]: Chest/Bust up.
    - [Object]: A specific item taking up 30-50% of frame (e.g., a laptop, a vase).

  4. 特写细节 (Detail View)
  - `close-up` (CU)
    - [Human]: Head/Face only.
    - [Object]: A specific small object fills frame (e.g., a phone, a coffee mug, a gun).
  - `extreme-close-up` (ECU)
    - [Human]: Eyes, Mouth, Finger.
    - [Object]: Texture details, text on a label, mechanism of a watch.

`5. camera-perspective`
- Rule: Select one best fit. Combines Angle, Height, and Relation.
- Tags:
  Angle (Pitch)
  - `eye-level` (Neutral)
  - `high-angle` (Looking down)
  - `low-angle` (Looking up)
  - `overhead-shot` (Bird's eye / 90° down)
  - `dutch-angle` (Tilted horizon)

  Height (Vertical Position - From your Image)
  - `shoulder-level` (Camera at subject's shoulder height)
  - `hip-level` (Camera at waist/gun height)
  - `knee-level` (Camera very low but not on ground)
  - `ground-level` (Worm's eye / Camera on floor)

  Relational (From your Image)
  - `pov` (First person view)
  - `over-the-shoulder` (Behind subject A looking at B)
  - `over-the-hip` (Lower variation of OTS, framed by arm/hip)

`6. character-composition`
- Rule: Select one.
- Tags: `no-character`, `single-character`, `two-shot`, `three-shot` (From your Image), `group-shot` (4+ distinct people), `crowd-shot` (Mass of people), `body-part-only`

`7. visual-style`
- Rule: Select one.
- Tags: `live-action`, `animation-2d`, `anime-style`, `animation-3d-cgi`, `stop-motion`, `mixed-media`, `monochrome` (Black & white)

`8. focus-and-optics` (New Category)
- Rule: Select one.
- Tags:
  - `deep-focus` (Everything sharp)
  - `shallow-focus` (Bokeh background)
  - `soft-focus` (Dreamy/Hazy)
  - `tilt-shift` (Miniature effect - From your Image)
  - `fisheye-lens` (Strong barrel distortion)
  - `macro-shot` (Microscopic detail)

**OUTPUT FORMAT:**
Your response must be only a single, well-formatted JSON object. Do not include any other text.

Example 1: A meaningful, high-quality scene.
```json
{
  "content-assessment": "meaningful-content",
  "quality-flags": [],
  "spatial-domain": "outdoor",
  "scene-atmosphere": [
    "day",
    "golden-hour",
    "natural-light"
  ],
  "shot-scale": "long-shot",
  "camera-perspective": "eye-level",
  "character-composition": "two-shot",
  "visual-style": [
    "live-action"
  ],
  "depth-of-field": "deep-focus"
}
```

Example 2: A meaningful but low-quality scene.
```json
{
  "content-assessment": "meaningful-content",
  "quality-flags": [
    "watermark-or-logo"
  ],
  "spatial-domain": "indoor",
  "scene-atmosphere": [
    "night",
    "low-key"
  ],
  "shot-scale": "medium-shot",
  "camera-perspective": null,
  "character-composition": "single-character",
  "visual-style": [
    "live-action"
  ],
  "depth-of-field": "deep-focus"
}
```

Example 3: An unusable solid color frame.
```json
{
  "content-assessment": "solid-color-screen",
  "quality-flags": [],
  "spatial-domain": null,
  "scene-atmosphere": [],
  "shot-scale": null,
  "camera-perspective": null,
  "character-composition": null,
  "visual-style": [],
  "depth-of-field": null
}
```

Now, analyze the given image and generate the JSON output.