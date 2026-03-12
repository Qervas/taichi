import{_ as r}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-B6D78dDW.js";import{o,b as u,w as e,g as s,d,m as c,ad as l,v as m,x as f,T as a}from"./modules/vue-CYPF2tZc.js";import{I as g}from"./slidev/default-CyU-U69Y.js";import{u as _,f as y}from"./slidev/context-Bmtznf1K.js";import"./modules/unplugin-icons-Djx3TAIU.js";import"./index-BmIMtRI9.js";import"./modules/shiki-DXooxXci.js";const M={__name:"slides-phase9-rising-flood.md__slidev_10",setup(k){const{$clicksContext:i,$frontmatter:p}=_();return i.setup(),(h,n)=>{const t=r;return o(),u(g,m(f(a(y)(a(p),9))),{default:e(()=>[n[1]||(n[1]=s("h1",null,"代码架构",-1)),d(t,c({},{title:"",ranges:[]}),{default:e(()=>[...n[0]||(n[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"phase9/")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── run.py              # 主入口：加载建筑、注入循环、导出")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── config.py           # 所有参数的单一来源")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── solver.py           # MPMFluid 类：P2G/Grid/G2P + SDF + 注入")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── mesh_surface.py     # PLY → OBJ (SplashSurf / Python fallback)")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── mesh_particles.py   # NPZ → 连续洪水面 OBJ (density grid)")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── blender_render.py   # Blender Cycles 渲染管线 (713 行)")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── find_camera.py      # 多角度相机测试")]),l(`
`),s("span",{class:"line"},[s("span",null,"├── render_gizmo.py     # XYZ 坐标轴可视化渲染")]),l(`
`),s("span",{class:"line"},[s("span",null,"└── assets/")]),l(`
`),s("span",{class:"line"},[s("span",null,"    ├── 1.glb           # 建筑模型（带地面）")]),l(`
`),s("span",{class:"line"},[s("span",null,"    ├── 1_noground.glb  # 建筑模型（无地面，渲染用）")]),l(`
`),s("span",{class:"line"},[s("span",null,"    └── 2.glb           # 备选建筑")])])],-1)])]),_:1},16),n[2]||(n[2]=s("div",{class:"mt-4 text-gray-400 text-sm"},[s("p",null,[s("strong",null,"设计原则：")]),s("ul",null,[s("li",null,[s("code",null,"config.py"),l(" 集中管理所有参数 — 修改一处全局生效")]),s("li",null,[s("code",null,"solver.py"),l(" 自包含 — 包含 MLS-MPM 核心 + SDF 加载 + 注入逻辑")]),s("li",null,"渲染和仿真完全解耦 — 通过 PLY/NPZ 文件交换数据")])],-1))]),_:1},16)}}};export{M as default};
