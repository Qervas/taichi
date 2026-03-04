import{_ as r}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-t_i6bOSH.js";import{o,b as u,w as a,g as s,ad as l,d,m as c,v as m,x as g,T as e}from"./modules/vue-BOLmPnWG.js";import{I as f}from"./slidev/default-0H2Mrhhh.js";import{u as k,f as h}from"./slidev/context-D47w-2Wi.js";import"./modules/unplugin-icons-BmlxQJto.js";import"./index-CpeWIp55.js";import"./modules/shiki-DqztgEPn.js";const M={class:"grid grid-cols-2 gap-8 mt-6"},L={__name:"slides-phase2.md__slidev_3",setup(_){const{$clicksContext:i,$frontmatter:t}=k();return i.setup(),(x,n)=>{const p=r;return o(),u(f,m(g(e(h)(e(t),2))),{default:a(()=>[n[2]||(n[2]=s("h1",null,"解决方案：MLS-MPM",-1)),n[3]||(n[3]=s("div",{class:"mt-4 text-center text-xl"},[s("p",null,[l("关键转变："),s("strong",null,"Pressure Poisson 求解 → 状态方程 (EOS)")]),s("p",null,"每个粒子自己算压力，不需要全局求解器")],-1)),s("div",M,[n[1]||(n[1]=s("div",null,[s("h3",null,"Material Point Method"),s("ul",null,[s("li",null,"1994 年 Sulsky 等人提出"),s("li",null,"Disney 用 MPM 做了《冰雪奇缘》的雪"),s("li",null,[l("2018 年 Hu et al. 提出 "),s("strong",null,"MLS-MPM"),l("——简化版")])]),s("h3",null,"核心思想"),s("p",null,[l("粒子 = "),s("strong",null,"material points（质点）")]),s("p",null,"每个粒子代表一小团材料：质量、体积、速度、内部状态"),s("p",null,[l("通过 "),s("strong",null,"grid"),l(' 互相"交流"')])],-1)),s("div",null,[d(p,c({},{title:"",ranges:[]}),{default:a(()=>[...n[0]||(n[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null," 传统方法的问题：")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null," Euler (grid)：")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ 容易求解方程")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ❌ numerical diffusion")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ❌ 拓扑变化困难")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null," Lagrange (particles)：")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ 精确追踪材料")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ❌ mesh tangling")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ❌ 大变形困难")]),l(`
`),s("span",{class:"line"},[s("span")]),l(`
`),s("span",{class:"line"},[s("span",null," MPM (hybrid)：")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ grid 求解 + particle 追踪")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ 任意大变形")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ 自动处理拓扑变化")]),l(`
`),s("span",{class:"line"},[s("span",null,"   ✅ 统一框架处理各种材料")])])],-1)])]),_:1},16)])])]),_:1},16)}}};export{L as default};
