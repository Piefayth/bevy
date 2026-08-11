# Retained UI design record

This document records the research and design constraints for a damage-tracked
retained renderer for Bevy UI. It is a working contract for the implementation,
not a description of code that already exists.

The objective is stricter than ordinary "retained mode":

- an unchanged UI entity is not extracted again;
- unchanged paint commands and GPU data are retained;
- pixels are repainted only when a complete dependency proof says their current
  value may be wrong;
- layout, placement, paint, raster, and composition changes are independent;
- a static UI performs no main-world UI work, no extraction, and no raster work;
  if the game renders a fresh world frame, only the irreducible composition of
  visible cached UI remains;
- correctness never depends on a timing threshold, heartbeat, or cache
  promotion heuristic.

## Scope decision

Bevy 0.19 deliberately separates `bevy_ui` from `bevy_ui_render`. The
`ui_api`/`ui_bevy_render` features and the separate `UiPlugin` and
`UiRenderPlugin` make an alternative renderer an intended use case. See
[PR #18703](https://github.com/bevyengine/bevy/pull/18703).

The first implementation is therefore an external replacement for
`bevy_ui_render`. Crate scope expands only when a tested requirement cannot be
implemented through public scheduling and render-graph APIs.

Known likely expansion points are:

1. `bevy_ui`: stock layout, stack construction, and clipping walk static trees.
   [Issue #22909](https://github.com/bevyengine/bevy/issues/22909) describes the
   same idle-cost and root-invalidation problem. We will first test whether an
   external plugin can gate the public `UiSystems` sets and handle placement
   separately.
2. `bevy_core_pipeline`: a fresh world frame still needs cached UI composited.
   We will first test a final-writer interposition that preserves Bevy's output
   attachment and presentation bookkeeping, avoiding a core-pipeline fork.

Current upstream UI-render work retains intermediate render data, but not final
pixels:

- [PR #24893](https://github.com/bevyengine/bevy/pull/24893): retained extracted
  render entities;
- [PR #25290](https://github.com/bevyengine/bevy/pull/25290): retained phase
  items;
- [PR #25191](https://github.com/bevyengine/bevy/pull/25191): retained vertex
  allocation;
- [goal #25149](https://github.com/bevyengine/bevy/issues/25149): overall UI
  rendering improvement goal.

These are compatible foundations, but they do not implement raster retention or
damage repair.

### Scope probe results

An external plugin can attach a run condition to a `UiSystems` set after the
systems in that set were registered. The red/green regression test is
`bevy_ui_render_retained/tests/ui_systems_scope.rs`. This is enough to stop all
layout work for a wholly quiet tree without changing `bevy_ui`.

It is not enough to separate layout from placement. `ui_layout_system`
currently performs both Taffy layout and the recursive update of
`ComputedNode`/`UiGlobalTransform`, including transforms and scroll offsets.
That recursive update is not a separately schedulable public system, and the
`UiSurface` holding the Taffy state is private. Therefore
a transform-only or scroll-only animation can avoid layout only by either:

- moving that geometry update into an independently scheduled `bevy_ui`
  system; or
- duplicating Bevy's geometry traversal in the third-party crate.

The second choice creates two owners for the same derived state and is rejected.
Unless a smaller public composition point appears during implementation,
placement-domain quiescence justifies expanding scope to `bevy_ui`. Static-tree
quiescence alone does not.

There is a second possible `bevy_ui` boundary for O(changes) candidate
nomination. In Bevy 0.19, lifecycle hooks run for component insertion,
replacement, and removal, but not for ordinary writes through `Mut<T>`. The
executable proof is `bevy_ui_render_retained/tests/mutation_scope.rs`.

Keeping stock components and `Changed<T>` is nevertheless an exact pull model,
not a correctness compromise. It nominates changed entities without
re-extracting unchanged entities, but its filters scan every matching entity.
On this development machine a quiet release-mode query over 10,000 entities
measured approximately 10--12 microseconds for one `Changed` input and 29--31
microseconds for an eight-input `Or`, including roughly 1.3 microseconds of
schedule overhead. At 1,000 entities the eight-input case measured about 5
microseconds. These are short local comparison points, not portable claims; the
source is `benches/benches/bevy_ui/changed.rs`.

The real constraint is dependency completeness. Canonical paint comparison
prevents a falsely nominated entity from causing damage, but cannot recover an
entity that a missing `Changed<T>` input never nominated. Each paint family
must therefore declare its complete component, asset, resource, removal, and
cross-entity dependencies next to the code that builds its canonical record,
with a mutation-matrix test for every declared input. A mutable dereference that
writes an equal value is a harmless false nomination: record comparison pays
the extraction for that entity but produces no upload or damage.

Accepting the measured O(matching entities x tested inputs) scan keeps the
renderer third-party and avoids changing Bevy's component mutation model. If
that scan is later rejected, the structurally correct alternatives are an ECS
mutation journal or immutable render-affecting values replaced through an
invalidating API. Manual dirty flags and periodic audits remain rejected
because they permit permanent stale pixels.

The core-pipeline probe found a public and simpler interposition point than
changing `CameraOutputMode` after target preparation. Bevy schedules the final
`upscaling` writer as an ordinary system in `Core2d`/`Core3d`, and the public
schedule API can remove a system by its function type. A third-party plugin can
therefore remove that one system and install its own final writer after
`Core2dSystems::PostProcess`/`Core3dSystems::PostProcess`. The proof in
`bevy_ui_render_retained/tests/core_pipeline_scope.rs` removes exactly one stock
writer and exactly one replacement.

This preserves Bevy's `ViewTarget`, output attachment, and later presentation
bookkeeping without vendoring `bevy_core_pipeline`. The scheduling proof does
not establish that the replacement blit has correct pixels or that every
window backend presents it; image-target readback will establish the former,
while the latter still requires a window integration test on each supported
backend.

Replacing individual stock extraction systems exposed one smaller
`bevy_ui_render` composition boundary. `Schedule::remove_systems_in_set` first
initializes a changed schedule, but `MainWorld` exists in the render world only
while `ExtractSchedule` is running. A third-party plugin therefore cannot
safely remove a stock extractor during plugin construction. Keeping the old
system behind a false run condition or temporarily moving the main world were
rejected.

The single-crate patch is `UiRenderInfrastructurePlugin`. It owns Bevy's UI
shader, camera extraction, phase, queue, batching, and buffer preparation but
no paint extraction or UI pass. Stock `UiRenderPlugin` composes that
infrastructure with its existing extractors and pass; the retained crate uses
the infrastructure directly. This preserves a maintainable third-party
renderer while requiring a small patch only to the crate it replaces. No
`bevy_ecs` or `bevy_core_pipeline` fork is required.

## Model learned from other UI systems

Mature UI systems retain several representations rather than one:

```text
application / ECS
    structure and state dependencies
layout tree
    measure and placement
paint artifact
    display commands and stable paint identity
raster surfaces
    physical coverage and pixel damage
compositor scene
    transforms, opacity, clips, scroll, and surfaces
presented buffers
    logical frame damage plus recycled-buffer history
```

Each transition has its own dirty domain. A single dirty bit cannot express the
minimum correct work.

### Flutter

Flutter separates layout, paint, compositing-bit, and composited-layer updates.
A repaint boundary gives a subtree a separate display list; an engine layer can
be retained across scenes; raster caching is an additional optimization rather
than the definition of retention.

- [`RenderObject.markNeedsPaint`](https://api.flutter.dev/flutter/rendering/RenderObject/markNeedsPaint.html)
- [`markNeedsCompositedLayerUpdate`](https://api.flutter.dev/flutter/rendering/RenderObject/markNeedsCompositedLayerUpdate.html)
- [`RenderRepaintBoundary`](https://api.flutter.dev/flutter/rendering/RenderRepaintBoundary-class.html)
- [`SceneBuilder.addRetained`](https://api.flutter.dev/flutter/dart-ui/SceneBuilder/addRetained.html)

Flutter also distinguishes this frame's damage from damage already present in a
recycled framebuffer. Its Metal backend accumulates damage separately for each
drawable texture. This is the exact solution to multi-buffer persistence; a
fixed "settling" duration is not.

- [Flutter rasterizer damage](https://github.com/flutter/flutter/blob/master/engine/src/flutter/shell/common/rasterizer.cc)
- [Metal drawable damage history](https://github.com/flutter/flutter/blob/master/engine/src/flutter/shell/gpu/gpu_surface_metal_skia.mm)

### Chromium

Chromium retains stable display items and paint chunks, then associates them
with four property trees: transform, clip, effect, and scroll. Raster
invalidation, compositor damage, and presentation damage are distinct. The
active/pending compositor trees preserve an old complete scene until a new one
is ready.

- [Blink paint artifacts and property trees](https://chromium.googlesource.com/chromium/src/%2B/HEAD/third_party/blink/renderer/platform/graphics/paint/README.md)
- [Compositor architecture](https://chromium.googlesource.com/chromium/src.git/%2B/master/docs/how_cc_works.md)
- [Compositor animations](https://chromium.googlesource.com/chromium/src/%2B/HEAD/third_party/blink/renderer/core/animation/README.md)

Appearing content damages its new bounds, disappearing content its old bounds,
and reordering/property changes generally damage old and new coverage. Transform,
opacity, scrolling, and some filters can be updated by the compositor without
rebuilding paint.

### Firefox WebRender

WebRender turns a display list into picture, spatial, and clip trees. Cached
picture tiles track all inputs that can affect them: primitives, clips, image
keys, opacity bindings, and transforms. Invalidated dependency leaves form an
exact tile dirty rectangle.

- [Firefox rendering overview](https://searchfox.org/firefox-main/source/gfx/docs/RenderingOverview.rst)
- [WebRender picture caching](https://searchfox.org/firefox-main/source/gfx/wr/webrender/src/picture.rs)

This validates explicit resource and property dependencies. WebRender also uses
heuristics to choose cache slices; those heuristics affect performance, not
correctness, and are not part of our initial design.

### Android Views and Jetpack Compose

Android `RenderNode` stores a display list separately from properties such as
translation, scale, rotation, and alpha. A content change rerecords the affected
node; a property animation does not. HWUI damages both the old and new node
state and joins historical damage according to buffer age.

- [Hardware-accelerated drawing model](https://developer.android.com/develop/ui/views/graphics/hardware-accel)
- [`RenderNode`](https://developer.android.com/reference/android/graphics/RenderNode)
- [HWUI buffer-age damage](https://android.googlesource.com/platform/frameworks/base/%2B/517a9583a869ed1913efeb003592db184d1a09e0/libs/hwui/renderthread/CanvasContext.cpp)

Compose tracks state reads by execution phase. Composition, measurement,
placement, and drawing have restart scopes. Reading an animated color during
drawing can restart drawing alone; reading an offset during placement can avoid
composition and measurement.

- [Compose phases](https://developer.android.com/develop/ui/compose/phases)
- [Compose phase performance](https://developer.android.com/develop/ui/compose/performance/phases)

This is the closest model for Bevy component-to-dirty-domain mapping.

### UIKit and Core Animation

Core Animation stores view content in layer backing bitmaps and maintains model,
presentation, and render layer trees. Cached content can be transformed and
faded without asking the view to redraw. Layout and display have separate dirty
APIs, and transactions commit layer-tree changes atomically.

- [Core Animation basics](https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/CoreAnimation_guide/CoreAnimationBasics/CoreAnimationBasics.html)
- [`CALayer`](https://developer.apple.com/documentation/quartzcore/calayer)
- [`CATransaction`](https://developer.apple.com/documentation/quartzcore/catransaction)

### Unity, Unreal, Qt, and WPF

Unity UI Toolkit retains its visual tree, vertex allocations, and batches. Its
`DynamicTransform`, `GroupTransform`, and `DynamicColor` usage hints avoid
unnecessary geometry regeneration. This demonstrates that retained geometry and
retained final pixels are separate features in a continuously rendered game.

- [Unity `UsageHints`](https://docs.unity3d.com/6000.0/ScriptReference/UIElements.UsageHints.html)
- [UI Toolkit performance](https://docs.unity3d.com/Manual/best-practice-guides/ui-toolkit-for-advanced-unity-developers/optimizing-performance.html)

Unreal makes that distinction explicit: invalidation caches hierarchy, layout,
and paint data, while a Retainer Panel additionally flattens a subtree into a
texture. Retainer repaint is expensive and consumes extra memory; volatile
widgets deliberately bypass paint caching.

- [Slate and UMG invalidation](https://dev.epicgames.com/documentation/unreal-engine/invalidation-in-slate-and-umg-for-unreal-engine)

Qt Quick synchronizes only changed items into a render-thread scene graph and
retains GPU geometry and batch roots. WPF similarly stores serialized retained
drawing data.

- [Qt Quick scene graph](https://doc.qt.io/qt-6/qtquick-visualcanvas-scenegraph.html)
- [Qt Quick renderer](https://doc.qt.io/qt-6/qtquick-visualcanvas-scenegraph-renderer.html)
- [WPF retained rendering](https://learn.microsoft.com/en-us/dotnet/desktop/wpf/graphics-multimedia/wpf-graphics-rendering-overview)

CSS containment demonstrates that a boundary must be a semantic promise. Layout
or paint containment permits optimization precisely because effects are
guaranteed not to escape the boundary.

- [CSS Containment](https://www.w3.org/TR/css-contain-2/)

## Dirty domains

The implementation uses these conceptual domains:

| Domain | Examples | Required work |
| --- | --- | --- |
| Structure | children, visibility participation, render family | rebuild affected identities/order/dependencies |
| Measure | `Node`, content size, text metrics, target scale | run layout for the affected dependency root |
| Placement | `UiTransform`, scroll offset | update spatial properties and affected descendant coverage |
| Paint | colors, glyphs, borders, image selection | rebuild only affected paint records and repair their pixels |
| Composite | retained-island transform or opacity | update compositor properties, no island repaint |
| Resource | image/atlas/material/sampled-surface generation | repaint exact readers and their filter reach |

For existing Bevy components, the intended semantics are:

- `Node` changes layout;
- `UiTransform` changes placement but not measurement;
- `BackgroundColor`, `BorderColor`, and equivalent visual components change
  paint but not layout;
- `ZIndex`, `GlobalZIndex`, hierarchy, and render-family changes affect order;
- a declared retained boundary allows subtree transform and opacity to become
  composite-only changes.

## Retained render representation

Every drawable has a stable `PaintId`. Its retained paint record contains every
input the renderer consumes:

- family-specific command data;
- paint order;
- transform, clip, and effect property references;
- exact conservative physical-pixel coverage;
- GPU allocation/batch ownership;
- sampled resource IDs, generations, and read regions.

Change notification only queues possible work. When a queued entity is
extracted, its canonical new paint records are compared with the retained old
records. Equal records do nothing. Changed records damage old union new
coverage.

We do **not** scan or fingerprint every extracted item each frame. That would be
correct but would violate the extraction objective. A full reference extraction
is retained only as a test oracle.

Custom draw families must declare complete coverage and resource dependencies.
The safe fallback for an incomplete contract is volatility: repaint it whenever
its containing view renders. Under-declared dependencies are never accepted as
an optimization.

## Raster surfaces and repair

A window needs a UI-owned persistent surface because its presentation textures
rotate. An image render target persists when its entire camera is inactive, as
proved by
`stock_image_target_persists_only_when_the_whole_camera_is_inactive`. Skipping
only Bevy's UI pass is insufficient: the remaining Core2d/upscaling work
overwrites the target with transparent pixels. A retained offscreen camera must
therefore become inactive after a completed repair, or use a render graph that
does no target write on a quiet frame.

For each damaged region:

1. Preflight every pipeline, bind group, texture, and buffer needed by the
   repair.
2. Keep old retained state and damage owed if preflight fails.
3. Wipe the damaged region with blending disabled.
4. Replay every intersecting record bottom-up, with the repair scissor active.
5. Commit the new retained state only after commands for the complete repair
   were encoded.

The region representation is a canonical union of integer physical rectangles.
No subpixel tolerance is required. A record's conservative coverage includes
antialiasing, shadows, outlines, filters, and texture sampling reach. Old
coverage remains authoritative until the new pixels commit, so subpixel drift
cannot accumulate invisibly.

Offscreen surfaces carry monotonically increasing content generations. Readers
declare the sampled surface and source region. A source commit invalidates the
mapped reader coverage exactly. This replaces fixed settling frames and
heartbeats.

Surface loss, resize, format/scale changes, atlas eviction, or device-resource
loss invalidate the affected retained surface completely.

## Composition

If the world below the UI changes, cached UI must be composited again. This is
irreducible unless the operating-system compositor owns the UI as a separate
surface, which portable `wgpu` does not expose.

Composition need not sample a fullscreen transparent layer. The renderer can
retain occupied coverage tiles or regions and issue a static instanced draw for
only nonempty UI coverage in the same render pass as Bevy's final world blit:

```text
draw world/upscaled image
draw cached occupied UI regions
```

This avoids an additional attachment load/store and makes sampling cost
proportional to visible UI coverage. A fullscreen translucent UI remains
fullscreen work by definition.

The public `wgpu::Surface` presentation API has no damage-region parameter, so
portable platform partial-present behavior cannot currently be promised.

- [`wgpu::Surface`](https://docs.rs/wgpu/latest/wgpu/struct.Surface.html)

## Handoff audit

The supplied prior implementation report is evidence, not specification.

The local `bevy-ui-retain` branch was audited as another evidence source. It
contains useful stable owner/arena allocation work, but nominates correctness
through a long explicit `Changed<T>` and `RemovedComponents<T>` inventory and
does not retain final pixels. Its code is not used as the new foundation. The
post-0.19 upstream PR stack is also not cherry-picked because it includes broad
render API churn beyond UI. The implementation keeps the compatible idea—a
stable retained allocation—while making candidate nomination subordinate to
canonical paint-record comparison.

Retained as invariants:

- premultiplied layer composition;
- old union new coverage for moved, removed, or reordered content;
- wipe then rebuild damaged pixels bottom-up;
- stable identity independent of transient list indices;
- damage remains owed until a complete repair is encoded;
- offscreen readers are explicit dependencies;
- text coverage composes glyph-local translation with node transforms;
- GPU readback, red-first regression tests, and retained-vs-full differential
  tests are the correctness instruments;
- declared repaint boundaries for content whose placement animates while its
  pixels remain stable.

Rejected from the initial design:

- a `0.5px` movement threshold;
- a fixed three-frame consumer settling period;
- a periodic heartbeat repaint;
- correctness based on a handwritten `Changed<T>` list alone;
- region-count, area, or promotion thresholds that create behavior cliffs;
- clearing dirty GPU uploads before the upload path is known to be available.

Still requiring measurement rather than assumption:

- fullscreen texture versus sparse/tiled surface storage on each target GPU;
- mobile tile-memory load/store behavior for the exact render-pass shape;
- whether folding UI composition into Bevy's final pass materially changes
  power consumption;
- the best deterministic tile size, if tiled storage is implemented.

## Verification contract

Every correctness feature lands with an automated proof where the platform API
allows it:

- unit tests for canonical rectangle unions, coverage mapping, stable identity,
  dependency propagation, order changes, removals, and owed damage;
- schedule tests proving static systems and extraction do not run;
- GPU readback tests for blending, wipe/repair, old/new movement, clipping,
  text transforms, sampled surfaces, resize, and resource unavailability;
- a deterministic adversarial scene rendered once with retention and once with
  full redraw, followed by an exact pixel comparison;
- counters asserted by tests: entities extracted, paint records changed,
  elements replayed, damaged pixels, surfaces repaired, and composite coverage.

Every regression test must first fail with the named defect reintroduced.

Performance is tested continuously, not inferred from the correctness suite.
Bevy's repository-wide benchmark crate uses Criterion and named baselines; the
retained renderer adds a `ui` benchmark target there and follows the existing
`cargo bench --bench <name> -- --save-baseline <baseline>` workflow. There was
no dedicated Bevy UI benchmark target when this work began, although the
benchmark crate already depended on `bevy_ui`.

Every incremental capability has three benchmark states over identical UI
trees:

- quiet: no UI input changes;
- localized: one leaf changes in exactly one dirty domain;
- full: every relevant leaf changes.

Wall-clock samples catch constant-factor regressions. Deterministic counters
catch complexity regressions even when timing noise is high. Benchmarked work
counts include roots and entities visited, entities extracted, paint records
compared/changed, bytes uploaded, records replayed, damaged physical pixels,
surface repairs, composite instances, and composite pixel coverage. The quiet
case is required to report zero for every counter except unavoidable
composition when another renderer produces a fresh target frame.

- [Bevy benchmark instructions](../benches/README.md)

The initial short-run baseline on 2026-08-10 used the stock `UiPlugin` over a
flat tree. At 10,000 nodes this machine measured approximately 1.18 ms quiet,
4.52 ms for one changed leaf, and 5.77 ms when every leaf changed. These are
local comparison points, not portable claims. More importantly, the quiet
100/1,000/10,000 results scale from roughly 0.08/0.17/1.18 ms, confirming the
static stock path remains linear in tree size. The benchmark source is
`benches/benches/bevy_ui/layout.rs`.

The first retained-core benchmark on the same machine measured quiet repair
planning at roughly 6 ns and one canonical record change plus exact damage at
roughly 0.26 microseconds, invariant across 100/1,000/10,000 retained records.
A full 10,000-record change took roughly 1.73 ms. This benchmark covers only the
canonical record map and exact damage journal; it does not yet include Bevy
extraction, GPU upload, raster repair, or composition.

After exact multi-region coverage was added, Criterion caught an avoidable
full-change regression to roughly 1.97 ms: `upsert` cloned both coverage sets.
Recording through disjoint borrows removed those copies. Replacing a four-rect
inline array—which enlarged every hash-map record—with compact
`Empty | One | Many` storage then measured roughly 2 ns quiet, 0.27
microseconds for one localized change, and 1.63 ms for 10,000 full changes in a
short Criterion run. The common zero/one-region cases allocate nothing.

The GPU acceptance harness uses a 64-by-64 image target, synchronous pipeline
compilation, explicit device polling, an in-process mutex, and a cross-process
file lock. Its stock and retained apps run in one process. A directly
constructed stock final scene is byte-identical to the retained scene reached
through movement, recoloring, and removal, including translucent overlap.

The first integrated retained families are backgrounds (`BackgroundColor` and
`OuterColor`) and ordinary, unsliced `ImageNode`s. They share one scene, one
stable `(entity, family, ordinal)` identity space, and one damage journal per
camera. Family extractors only update canonical records; a single later replay
stage sorts every visible family together. This is required for correctness:
repairing a translucent image must first replay the background beneath it.
The named GPU proof changes a 10-by-10 image, repairs exactly 100 pixels, and
replays exactly those two intersecting records bottom-up.

Records use `Changed<T>` candidate nomination, bit-exact canonical values,
stable render entities, and old-union-new damage. Each exact, non-overlapping
damage rectangle is wiped to transparent and rebuilt from only the sorted
phase items whose physical bounds intersect it. A localized background proof
moves a 10-by-10 leaf across a larger background: exactly 200 pixels are
repaired and only the three necessary item-region intersections are replayed.

The mutation audit found that `ComputedNode` alone is not a complete
nomination source: Bevy intentionally writes resolved borders and corner radii
through `bypass_change_detection()`. `Changed<Node>` and
`Changed<ComputedUiRenderTargetInfo>` therefore nominate those derived paint
inputs; canonical comparison still decides whether the record and pixels
actually changed. A red-first GPU test changes only the source border while
the node size and transform remain fixed. Retargeting to an entity without a
render camera also removes the old camera's records immediately instead of
leaving stale ownership behind.

The current background dependency matrix is executable in `tests/gpu_ui.rs`:

| Pixel dependency | Nomination or lifecycle source | Named proof |
| --- | --- | --- |
| fill color | `Changed`/removed `BackgroundColor` | equal write, real color change, adversarial despawn |
| outer rounded-corner fill | `Changed`/removed `OuterColor` | `removing_outer_color_repairs_its_vacated_pixels` |
| resolved size/border/radius | `Changed<ComputedNode>`, plus source `Node` and target info | movement and `source_node_changes_nominate_bypassed_computed_paint_geometry` |
| target-space transform | `Changed<UiGlobalTransform>` | `ui_transform_motion_nominates_only_the_moved_leaf` |
| inherited visibility | `Changed`/removed `InheritedVisibility` | `inherited_visibility_removes_only_the_hidden_leaf_pixels` |
| calculated clip | changed, inserted, or removed `CalculatedClip` | named clip insertion and removal tests |
| target camera ownership | `Changed`/removed `ComputedUiTargetCamera` | `losing_a_renderable_target_removes_pixels_from_the_previous_camera` |
| paint order | `Changed`/removed `ComputedStackIndex` | `computed_stack_changes_repair_translucent_paint_order` |
| UI participation | changed/removed source `Node` | `removing_node_ends_ui_participation_even_if_computed_components_remain` |
| viewport size and scale | `Changed<ComputedUiRenderTargetInfo>` plus surface mismatch | `resizing_a_viewport_reconstructs_its_retained_surface` |

All canonical draw fields remain in the fingerprint, so these sources only
nominate comparison. They do not decide damage.

Image resource dependencies are exact and reverse-indexed. `Image` asset
changes nominate only `ImageNode`s that sample that asset and add a content
generation to their canonical record. `TextureAtlasLayout` changes nominate
only nodes using that layout, then recompute the selected rectangle; changing
an unselected atlas entry is therefore compared but causes zero damage. The
GPU suite proves quiet images match stock output, image tint changes rebuild
underlying backgrounds, pixel-only asset modifications repaint exact readers,
selected atlas changes repaint, and irrelevant atlas edits do not.

Availability follows Bevy's render-asset lifecycle rather than main-world
`Assets<Image>` membership. Removing a main-world asset does not unload its GPU
copy while a strong handle remains, so it correctly causes no repaint. A
never-available image is safely omitted and its old coverage erased. An added
or modified image remains pending until `RenderAssets<GpuImage>` contains the
new generation; while a byte-upload budget deliberately holds it back, the
old retained pixels stay visible and the damage remains owed. Per-item batch
components are cleared before preparation so readiness can never be satisfied
by stale metadata from a previous frame.

Solid borders and outlines retain four fixed edge records per visible family.
This deliberately separates damage identity from GPU draw grouping. A first
attempt drew every edge independently; GPU comparison rejected it because
equal-color corner ties then alpha-blended more than once. The accepted path
keeps edge identity stable, but merges equal canonical commands at replay time
by OR-ing their border flags. Changing the left edge out of an equal-color
group therefore damages only that edge's 140-pixel rounded-corner reach, while
the other three edges still replay as one command under the damage scissor.
Equal-color borders, distinct-color regrouping, outlines, and component
removal are byte-identical to stock in named GPU tests.

Transparent or empty paint is represented by no retained record. This matters
because `Node` requires transparent background and border components: keeping
those as invisible records would still compare and rewrite canonical state
whenever an unrelated transform moved. A GPU/counter proof moves an unpainted
node and observes zero paint candidates, records, surfaces, or repairs.

A paint record owns an exact set of physical coverage rectangles rather than
one bounding box. Its compact `Empty | One | Many` representation allocates
nothing for ordinary records; genuinely disjoint sampled-image mappings pay
for storage only when they use it. Damage, item intersection, and cached
composite coverage all consume the same region set. A unit proof keeps the gap
between two regions absent from damage.

The owned UI layer is double-buffered. A repair encodes into the inactive
texture and replaces the visible texture only after every region succeeds.
Each slot carries a committed generation; bringing an older slot current
copies only damage committed since that generation, rather than copying the
whole layer. Failed or not-yet-compiled work remains owed. On a quiet
background-only frame, GPU counters prove zero additional repairs, repair
pixels, and replayed items while composition continues. An equal component
replacement nominates and compares exactly one record but changes zero records
and causes zero repairs; a real color change repairs exactly that record's old
and new physical coverage.

Each layer is sized in viewport-local physical pixels. UI repair uses that
local surface directly; only composition applies the camera viewport. A
nonzero-origin viewport is byte-identical to stock, and moving an unchanged
viewport is proven to cause zero paint candidates and zero repair work. This
is placement-only motion of a retained surface.

Composition caches the exact union of possibly nontransparent item bounds when
a repair commits. Quiet frames reuse those regions and issue one scissored
texture draw per non-overlapping region; they do not rebuild the union. A lone
10-by-10 UI composites 100 pixels rather than the 4,096-pixel test viewport.
Two disjoint translucent 10-by-10 regions remain byte-identical to stock,
preserve the untouched gap, and composite exactly 200 pixels in two draws. An
empty committed layer stops compositing entirely, and a camera with no visible
UI allocates no layer textures. If a layer is lost or no longer matches its
viewport size or target format while its records remain unchanged, the whole
viewport is added to owed damage and those records replay on the following
extraction. Replacing the scissored draws with one static instanced region draw
remains a constant-factor optimization; it does not change pixel coverage or
invalidation semantics.

Some behavior cannot be established by portable automated tests:

- DRAM traffic, tile-memory behavior, energy use, and battery impact require
  profiling on each physical GPU architecture;
- operating-system compositor behavior and partial presentation are hidden by
  `wgpu`;
- device/driver command-buffer failures may only be observable through
  backend-specific completion diagnostics.

These are reported as measured platform results, never inferred from desktop
timings. Pixel correctness remains testable through image targets and device
readback even when the performance mechanism is opaque.

## Increment order

1. Prove external `UiSystems` quiescence and final-writer interposition. These
   decide crate scope.
2. Build a serialized GPU-test harness and retained-vs-full differential scene.
3. Retain canonical paint records and process only changed/removed entities.
4. Add a persistent window UI surface, full repair on change, and complete skip
   on a quiet frame.
5. Add exact damage regions, wipe/replay, old/new coverage, and per-record
   culling.
6. Add transform/clip/effect/scroll property separation and declared retained
   boundaries.
7. Add exact sampled-surface/resource dependency generations.
8. Gate or replace main-world layout, stack, and clipping work by dirty domain.
9. Perform runtime-selectable on-device architecture sweeps and retain only
   mechanisms justified by measurements.

Cleanup is part of every increment: superseded paths, flags, thresholds, and
comments are removed before the next capability is added.
