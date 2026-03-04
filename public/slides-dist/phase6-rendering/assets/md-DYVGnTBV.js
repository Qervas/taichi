import{_ as d}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-DBQLO8Mr.js";import{o as u,b as c,w as a,g as s,d as t,m as i,ad as l,v as m,x as g,T as p}from"./modules/vue-DJcJNTRv.js";import{I as f}from"./slidev/default-B1UWz_Wk.js";import{u as k,f as h}from"./slidev/context-DIL9icuY.js";import"./modules/unplugin-icons-C1PhRCkS.js";import"./index-BgUpo8iM.js";import"./modules/shiki-CnwWDitJ.js";const v={class:"grid grid-cols-2 gap-8 mt-6"},x={class:"p-4 bg-red-900 bg-opacity-30 rounded"},b={class:"p-4 bg-green-900 bg-opacity-30 rounded"},N={__name:"slides-phase6-rendering.md__slidev_15",setup(_){const{$clicksContext:r,$frontmatter:o}=k();return r.setup(),(M,n)=>{const e=d;return u(),c(f,m(g(p(h)(p(o),14))),{default:a(()=>[n[4]||(n[4]=s("h1",null,"D. 为什么 MLS-MPM 不适合洪水仿真",-1)),n[5]||(n[5]=s("div",{class:"mt-6 text-center text-2xl"},[s("p",null,[s("strong",null,"核心问题：维度失配")])],-1)),s("div",v,[s("div",x,[n[1]||(n[1]=s("h3",null,"MLS-MPM（3D 方法）",-1)),t(e,i({},{title:"",ranges:[]}),{default:a(()=>[...n[0]||(n[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"500m × 500m × 30m 域")]),l(`
`),s("span",{class:"line"},[s("span",null,"网格: 640 × 640 × 40")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null,"= 16,400,000 个 3D 网格节点")]),l(`
`),s("span",{class:"line"},[s("span",null,"≈ 8,000,000 个粒子")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null,"绝大部分垂直节点里是空气")]),l(`
`),s("span",{class:"line"},[s("span",null,"→ 做了大量无用功")])])],-1)])]),_:1},16)]),s("div",b,[n[3]||(n[3]=s("h3",null,"SWE（2D 方法）",-1)),t(e,i({},{title:"",ranges:[]}),{default:a(()=>[...n[2]||(n[2]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"500m × 500m 域")]),l(`
`),s("span",{class:"line"},[s("span",null,"网格: 512 × 512")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null,"= 262,144 个 2D 格子")]),l(`
`),s("span",{class:"line"},[s("span",null,"每格存 3 个标量 (h, hu, hv)")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null,"每个格子都在做有用的计算")]),l(`
`),s("span",{class:"line"},[s("span",null,"→ 零浪费")])])],-1)])]),_:1},16)])]),n[6]||(n[6]=s("div",{class:"mt-4 text-center"},[s("p",null,[s("strong",null,"状态量之比：16.4M vs 262K → SWE 少 63 倍")])],-1))]),_:1},16)}}};export{N as default};
