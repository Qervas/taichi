import{_ as t}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-BI9G3k5u.js";import{o,b as u,w as a,g as s,d as c,m as d,ad as n,v as y,x as g,T as e}from"./modules/vue-CYPF2tZc.js";import{I as m}from"./slidev/default-uxzad39v.js";import{u as f,f as _}from"./slidev/context-Bgcaj8gG.js";import"./modules/unplugin-icons-Djx3TAIU.js";import"./index-ClRhfRrj.js";import"./modules/shiki-DXooxXci.js";const b={class:"text-sm mt-2"},L={__name:"slides-phase8-building-flood.md__slidev_10",setup(v){const{$clicksContext:p,$frontmatter:i}=f();return p.setup(),(h,l)=>{const r=t;return o(),u(m,y(g(e(_)(e(i),9))),{default:a(()=>[l[1]||(l[1]=s("h1",null,"模块化代码架构",-1)),s("div",b,[c(r,d({},{title:"",ranges:[]}),{default:a(()=>[...l[0]||(l[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"phase8/")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── config.py              # 场景配置 (单一真相源)")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── run.py                 # 独立版 (单一 MPM, 含水粒子)    717 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── run_hybrid.py          # 混合版入口                       38 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"│")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── solver/                # 混合求解器包")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── __init__.py        # exports: Solver, HybridSolver")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── hybrid.py          # 编排层: SWE + MPM + 耦合         152 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── swe.py             # 2D 浅水方程 (HLLC/MUSCL)         434 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── engine.py          # 3D MLS-MPM (固体)                605 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── fields.py          # Taichi 场分配")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── materials.py       # 本构模型 (无 Taichi 依赖)")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ├── coupling.py        # SWE→MPM 力传递 + 浸泡损伤        268 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   └── colliders.py       # 建筑/车辆碰撞几何")]),n(`
`),s("span",{class:"line"},[s("span",null,"│")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── fracture.py            # Blender 建筑体素化脚本            273 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── blender_render.py      # Blender 渲染脚本                  931 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── scene_setup.py         # 场景初始化                        368 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"├── export.py              # PLY + NPZ 导出                    273 行")]),n(`
`),s("span",{class:"line"},[s("span",null,"└── visualize.py           # 可视化工具                        109 行")])])],-1)])]),_:1},16)]),l[2]||(l[2]=s("div",{class:"grid grid-cols-3 gap-3 mt-2 text-xs"},[s("div",{class:"p-2 bg-blue-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"config.py 是唯一配置源"),n("：所有参数（域、分辨率、材料、车辆布局）都在这里。solver 和 renderer 都导入它。")])]),s("div",{class:"p-2 bg-green-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"run.py vs run_hybrid.py"),n("：run.py 是独立版（纯 MPM），run_hybrid.py 用 HybridSolver。两种模式共存。")])]),s("div",{class:"p-2 bg-purple-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"materials.py 无 Taichi 依赖"),n("：本构数学可以在 CPU 上测试，不需要 GPU。单元测试友好。")])])],-1))]),_:1},16)}}};export{L as default};
