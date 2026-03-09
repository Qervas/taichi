import{_ as r}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-BI9G3k5u.js";import{o as t,b as o,w as a,g as l,d,m as c,ad as n,v as g,x as m,T as e}from"./modules/vue-CYPF2tZc.js";import{I as f}from"./slidev/default-uxzad39v.js";import{u as h,f as k}from"./slidev/context-Bgcaj8gG.js";import"./modules/unplugin-icons-Djx3TAIU.js";import"./index-ClRhfRrj.js";import"./modules/shiki-DXooxXci.js";const _={class:"grid grid-cols-2 gap-6 mt-4"},D={__name:"slides-phase8-building-flood.md__slidev_12",setup(P){const{$clicksContext:u,$frontmatter:i}=h();return u.setup(),(B,s)=>{const p=r;return t(),o(f,g(m(e(k)(e(i),11))),{default:a(()=>[s[5]||(s[5]=l("h1",null,"Blender 渲染管线",-1)),l("div",_,[l("div",null,[s[1]||(s[1]=l("h3",null,"导出 → 渲染 流程",-1)),d(p,c({},{title:"",ranges:[]}),{default:a(()=>[...s[0]||(s[0]=[l("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[l("code",{class:"language-text"},[l("span",{class:"line"},[l("span",null,"Phase 8 仿真 (Taichi GPU)")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ├── SWE: swe_000000.npz (h, hu, hv)")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ├── MPM: solid_000000.ply (pos, damage, chunk_id)")]),n(`
`),l("span",{class:"line"},[l("span",null,"    └── Meta: scene_meta.json")]),n(`
`),l("span",{class:"line"},[l("span",null,"           │")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ┌──────┘")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ▼")]),n(`
`),l("span",{class:"line"},[l("span",null,"fracture.py (Blender)")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │  FBX → BVH 射线检测 → 粒子化")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │  Voronoi 种子 → chunk ID")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ▼")]),n(`
`),l("span",{class:"line"},[l("span",null,"blender_render.py (Blender)")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │  粒子 → Marching Cubes → Mesh")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │  水面 → 高度场 → Mesh + 材质")]),n(`
`),l("span",{class:"line"},[l("span",null,"    │  光照 + 材质 + 相机动画")]),n(`
`),l("span",{class:"line"},[l("span",null,"    ▼")]),n(`
`),l("span",{class:"line"},[l("span",null,"最终 PNG 序列 → FFmpeg → MP4")])])],-1)])]),_:1},16),s[2]||(s[2]=l("h3",null,"导出规模",-1)),s[3]||(s[3]=l("ul",null,[l("li",null,[l("strong",null,"900 帧"),n(" @ 30 FPS = 30 秒动画")]),l("li",null,"每帧：1 SWE NPZ + 1 Solid PLY + 1 Car NPZ"),l("li",null,[n("总量："),l("strong",null,"~15 GB"),n(" 导出数据")])],-1))]),s[4]||(s[4]=l("div",null,[l("h3",null,"blender_render.py 亮点"),l("p",null,[l("strong",null,"931 行"),n(" — 系列最大的渲染脚本")]),l("ol",null,[l("li",null,[l("p",null,[l("strong",null,"粒子 → 网格")]),l("ul",null,[l("li",null,"同 chunk 粒子做 Metaball / Marching Cubes"),l("li",null,"碎块有锐利边缘，不是球形")])]),l("li",null,[l("p",null,[l("strong",null,"SWE 水面")]),l("ul",null,[l("li",null,"h(x,y) 高度场 → 位移网格"),l("li",null,"半透明 Glass BSDF + Fresnel")])]),l("li",null,[l("p",null,[l("strong",null,"损伤着色")]),l("ul",null,[l("li",null,"D < 0.3: 灰白混凝土"),l("li",null,[n("D ∈ "),l("span",null,"0.3, 0.7"),n(": 暗灰裂纹")]),l("li",null,[n("D ∈ "),l("span",null,"0.7, 1.0"),n(": 锈色断面")]),l("li",null,"D ≥ 1.0: 碎块灰")])]),l("li",null,[l("p",null,[l("strong",null,"相机动画")]),l("ul",null,[l("li",null,"全景 → 推近 → 跟踪坍塌"),l("li",null,"多角度切换")])])])],-1))])]),_:1},16)}}};export{D as default};
