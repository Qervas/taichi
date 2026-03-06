/* ━━━━━━━━━━━━━━━━━━━━ STATIC DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
export const POSTS = [
  {
    id: 11,
    phase: "Survey",
    title: "AI Physics Simulation Survey",
    subtitle: "2024–2026 · Neural Operators · GNN · 3DGS+MPM · Video World Models",
    date: "2026-03-06",
    categories: ["slides"],
    description:
      "Comprehensive survey of 30+ papers on AI-driven physics simulation (2024–2026). Covers two main tracks: AI-accelerated solvers (FNO, GNS, Walrus, PhysiX, weather foundation models, Flow Matching) and video-to-physics parameter extraction (PAC-NeRF → GIC → OmniPhysGS, PhysDreamer, PhysTwin, V-JEPA). Includes glossary, trend analysis, and connections to our MLS-MPM work.",
    slides: "slides-dist/ai-physics-survey/index.html",
  },
  {
    id: 1,
    phase: "Phase 01",
    title: "Smoke Simulation",
    subtitle: "2D Eulerian — MAC Grid · MacCormack Advection · Red-Black Gauss-Seidel",
    date: "2026-02-24",
    categories: ["slides", "vids"],
    description:
      "Kick-off of the series. Builds a real-time 2D smoke simulation from scratch on a 512×512 MAC staggered grid. Covers MacCormack advection with RK3 backtracing, Red-Black Gauss-Seidel pressure projection, buoyancy, and vorticity confinement.",
    video: "videos/smoke_simulation.mp4",
    thumb: "videos/thumbs/smoke_simulation.jpg",
    slides: "slides-dist/phase1/index.html",
  },
  {
    id: 7,
    phase: "Phase 05",
    title: "Water Wheel",
    subtitle: "MLS-MPM + grid-velocity rigid coupling on a 64³ grid",
    date: "2026-03-02",
    categories: ["vids", "code"],
    description:
      "First rigid–fluid coupling experiment. A kinematic water wheel spins at fixed ω inside a 64³ MLS-MPM grid. After the P2G step, every grid node inside the wheel's volume has its velocity overridden with v = ω × r, forcing the surrounding water particles to respond to the rotating boundary — no explicit constraint forces needed, just a grid-velocity override.",
    video: "videos/water_wheel_1.mp4",
    thumb: "videos/thumbs/water_wheel_1.jpg",
    code: "code/phase5_wheel.py",
  },
  {
    id: 8,
    phase: "Seminar",
    title: "MLS-MPM Unified Seminar",
    subtitle: "Algorithm · Paper Math · Official Taichi Demos · Live Coding",
    date: "2026-03-03",
    categories: ["slides", "video"],
    description:
      "60-slide unified deck covering the full MLS-MPM pipeline: continuum mechanics foundations, paper math (Hu et al. SIGGRAPH 2018), line-by-line walkthrough of official Taichi demos (mpm88, mpm99, mpm3d_ggui), parameter tuning discussion, and Phase 5 water wheel extension. Includes annotated Chinese versions of mpm88.py and mpm99.py.",
    video: "videos/water_wheel.mp4",
    thumb: "videos/thumbs/water_wheel.jpg",
    slides: "slides-dist/seminar/index.html",
    refs: [
      { label: "Official mpm88.py", url: "https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm88.py" },
      { label: "Official mpm99.py", url: "https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm99.py" },
    ],
  },
  {
    id: 10,
    phase: "Phase 07",
    title: "City Flood Simulation",
    subtitle: "2D SWE + HLLC/MUSCL · Two-Way Car Coupling · Real-Time Preview",
    date: "2026-03-05",
    categories: ["slides", "vids", "code"],
    description:
      "2D city flood simulation from top view. Shallow water equations with HLLC Riemann solver and MUSCL reconstruction, two-way car-fluid coupling (water→car: buoyancy + drag + torque, car→water: volume displacement + momentum source), Manning friction, CFL adaptive timestep, and continuous inflow boundary. 30 cars on a 512×512 grid, 60s simulation on Taichi GPU.",
    video: "videos/phase7-city-flood.mp4",
    thumb: "videos/thumbs/phase7-city-flood.jpg",
    slides: "slides-dist/phase7-city-flood/index.html",
    code: "code/phase7",
    codeFiles: [
      "run.py", "car_coupling.py", "city_data.py",
    ],
  },
  {
    id: 9,
    phase: "Phase 06",
    title: "Flood Rendering Pipeline",
    subtitle: "SWE + HLLC/MUSCL · HDR Rendering · PBR Water Surface · MLS-MPM Comparison",
    date: "2026-03-04",
    categories: ["slides", "video", "code"],
    description:
      "Phase 6 rendering pipeline technical summary. Covers shallow water equation (SWE) solver with HLLC Riemann solver and MUSCL reconstruction, HDR rendering pipeline, PBR water surface shading, and a detailed comparison of why MLS-MPM is not suitable for large-scale flood simulation.",
    video: "videos/large_scale_water.mp4",
    thumb: "videos/thumbs/large_scale_water.jpg",
    slides: "slides-dist/phase6-rendering/index.html",
    code: "code/phase6",
    codeFiles: [
      "run.py", "renderer.py", "terrain.py",
      "shaders/water.frag", "shaders/water.vert",
      "shaders/terrain.frag", "shaders/terrain.vert",
      "shaders/sky.frag", "shaders/sky.vert",
      "shaders/post.frag", "shaders/post.vert",
    ],
  },
  {
    id: 6,
    phase: "Paper",
    title: "MLS-MPM Paper Breakdown",
    subtitle: "Hu et al. SIGGRAPH 2018 — Moving Least Squares Material Point Method",
    date: "2026-02-28",
    categories: ["slides"],
    description:
      "Deep reading of the original MLS-MPM paper (Hu et al., SIGGRAPH 2018). Covers the mathematical derivation of the MLS weight functions, the unified MPM formulation, the P2G/G2P transfer schemes, and why replacing APIC's B matrix with MLS gradients gives both accuracy and efficiency gains.",
    slides: "slides-dist/paper/index.html",
    refs: [
      { label: "GAMES201 Lecture — MLS-MPM", url: "https://yuanming.taichi.graphics/teaching/2020-games201/#:~:text=Least%20Squares%20MPM%2C-,MLS%2DMPM,-%EF%BC%89" },
      { label: "Paper — Hu et al. SIGGRAPH 2018", url: "https://yuanming.taichi.graphics/publication/2018-mlsmpm/" },
    ],
  },
  {
    id: 2,
    phase: "Demo",
    title: "MLS-MPM Code Walkthrough",
    subtitle: "Official Taichi mpm3d.py line-by-line + screen-space water rendering",
    date: "2026-02-27",
    categories: ["slides", "vids"],
    description:
      "Deep-dive into the official Taichi mpm3d.py. Covers the 3-step P2G→Grid→G2P loop, the EOS stress formula, the C matrix as velocity gradient, B-spline weights, and the screen-space fluid rendering pipeline (depth splat → bilateral blur → normal reconstruction → Fresnel + Beer's law).",
    videos: [
      { src: "videos/mlsmpm1.mp4", thumb: "videos/thumbs/mlsmpm1.jpg", label: "2D MLS-MPM" },
      { src: "videos/mlsmpm2.mp4", thumb: "videos/thumbs/mlsmpm2.jpg", label: "3D Bridge Destruction" },
    ],
    slides: "slides-dist/demo/index.html",
  },
  {
    id: 3,
    phase: "Phase 02",
    title: "2D Water Dam Break",
    subtitle: "Weakly compressible MLS-MPM with EOS pressure",
    date: "2026-02-25",
    categories: ["code", "vids", "slides"],
    description:
      "A 2D dam-break simulation on a 128×128 collocated grid with ~50 K water particles (4 per cell). Pressure is computed per-particle via a two-sided equation of state — p = K(1 − J) — resisting both compression and expansion. The APIC affine matrix C doubles as the velocity gradient, giving accurate angular-momentum conservation at zero extra cost.",
    video: "videos/water_simulation_2d.mp4",
    thumb: "videos/thumbs/water_simulation_2d.jpg",
    slides: "slides-dist/phase2/index.html",
    code: "code/phase2_water.py",
  },
  {
    id: 4,
    phase: "Phase 03",
    title: "3D Bridge Destruction",
    subtitle: "Drucker-Prager fracture with two-phase MPM coupling",
    date: "2026-02-26",
    categories: ["code", "vids", "slides"],
    description:
      "A wall of water impacts four bridge pillars on a 64³ grid (270 K particles: 248 K water + 21.6 K solid). Solid uses Drucker-Prager elastoplasticity with SVD-based return mapping. Damage accumulates as yield strain exceeds a threshold, progressively degrading cohesion and triggering cascading shattering. Water and solid couple automatically through the shared MPM grid — no explicit coupling terms needed.",
    video: "videos/water_simulation_3d_pillars.mp4",
    thumb: "videos/thumbs/water_simulation_3d_pillars.jpg",
    slides: "slides-dist/phase2/index.html",
    code: "code/phase3_3d.py",
  },
  {
    id: 5,
    phase: "Phase 04",
    title: "Flood vs. Concrete Bridge",
    subtitle: "River-scale simulation on an elongated 128×64×64 domain",
    date: "2026-02-26",
    categories: ["code", "slides"],
    description:
      "Extends Phase 3 to a full river-flood scenario on a 128×64×64 domain. Replaces Drucker-Prager with Rankine tensile failure for realistic concrete cracking. Adds pile foundation scour erosion — the #1 real-world cause of bridge collapse. Rubble switches to J-based EOS for hard-chunk behaviour instead of sand-like scatter.",
    slides: "slides-dist/phase2/index.html",
    code: "code/phase4_flood.py",
  },
]

export const STATS = [
  { k: "Framework", v: "Taichi Lang" },
]

export const CATS = [
  { id: "all",    label: "All"    },
  { id: "slides", label: "Slides" },
  { id: "code",   label: "Code"   },
  { id: "vids",   label: "Vids"   },
]

export const ICONS = {
  slides: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/></svg>',
  code:   '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  vids:   '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
  ext:    '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>',
  ref:    '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"/></svg>',
  download: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
}

export const DOW = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
export const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

export const POST_DATES = new Set(POSTS.map(p => p.date))

export const todayStr = (() => {
  const t = new Date()
  return t.getFullYear() + "-" + String(t.getMonth() + 1).padStart(2, "0") + "-" + String(t.getDate()).padStart(2, "0")
})()

export function fmtDate(ds) {
  const [y, m, d] = ds.split("-").map(Number)
  return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1] + " " + d + ", " + y
}
