import{_ as o}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-BI9G3k5u.js";import{o as r,b as c,w as a,g as s,d as u,m as d,ad as n,v as g,x as m,T as e}from"./modules/vue-CYPF2tZc.js";import{I as f}from"./slidev/default-uxzad39v.js";import{u as k,f as b}from"./slidev/context-Bgcaj8gG.js";import"./modules/unplugin-icons-Djx3TAIU.js";import"./index-ClRhfRrj.js";import"./modules/shiki-DXooxXci.js";const v={class:"text-sm mt-2"},M={__name:"slides-phase8-building-flood.md__slidev_4",setup(_){const{$clicksContext:i,$frontmatter:p}=k();return i.setup(),(h,l)=>{const t=o;return r(),c(f,g(m(e(b)(e(p),3))),{default:a(()=>[l[1]||(l[1]=s("h1",null,"A. HybridSolver 架构",-1)),s("div",v,[u(t,d({},{title:"",ranges:[]}),{default:a(()=>[...l[0]||(l[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"┌─────────────────────────────────────────────────────────┐")]),n(`
`),s("span",{class:"line"},[s("span",null,"│                    HybridSolver.step()                   │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│                                                          │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   ┌──────────────────┐    ┌───────────────────────────┐ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  SWE (256² 2D)   │    │    MPM (128³ 3D)          │ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  ─────────────── │    │  ────────────────────────  │ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  HLLC + MUSCL     │    │  P2G → Grid → [Hook] → G2P│ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  h, hu, hv fields│◄───│  Concrete + Car particles  │ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  Wall obstacles   │    │  Rankine fracture          │ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   │  CFL adaptive dt  │───►│  SWE force injection       │ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│   └──────────────────┘    └───────────────────────────┘ │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│          │                           │                    │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│          └─────────┐  ┌──────────────┘                    │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│                    ▼  ▼                                   │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│              Coupling Layer                               │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│     ┌─────────────────────────────┐                      │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│     │ Buoyancy + Drag + Wall Pres.│                      │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│     │ Soaking Damage (per frame)  │                      │")]),n(`
`),s("span",{class:"line"},[s("span",null,"│     └─────────────────────────────┘                      │")]),n(`
`),s("span",{class:"line"},[s("span",null,"└─────────────────────────────────────────────────────────┘")])])],-1)])]),_:1},16)]),l[2]||(l[2]=s("div",{class:"grid grid-cols-3 gap-3 mt-3 text-xs"},[s("div",{class:"p-2 bg-blue-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"SWE 先行"),n("：每帧先推进 SWE（CFL 子循环），水面场冻结后再跑 MPM。")])]),s("div",{class:"p-2 bg-green-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"Hook 注入"),n("：MPM 的 P2G→Grid 之后、G2P 之前，插入 SWE 力。")])]),s("div",{class:"p-2 bg-purple-900 bg-opacity-20 rounded"},[s("p",null,[s("strong",null,"浸泡损伤"),n("：每帧结束时，SWE 水位信息→所有浸泡混凝土粒子累积损伤。")])])],-1))]),_:1},16)}}};export{M as default};
