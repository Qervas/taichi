import{_ as o}from"./slidev/CodeBlockWrapper.vue_vue_type_script_setup_true_lang-usmQVMfu.js";import{o as r,b as c,w as a,g as s,d as u,m as d,ad as n,v as m,x as _,T as e}from"./modules/vue-CYPF2tZc.js";import{I as f}from"./slidev/default-B2HT5pdf.js";import{u as x,f as k}from"./slidev/context-lIuGYNTU.js";import"./modules/unplugin-icons-Djx3TAIU.js";import"./index-BATq-hrq.js";import"./modules/shiki-DXooxXci.js";const h={class:"mt-4"},M={__name:"slides-phase7-city-flood.md__slidev_8",setup(g){const{$clicksContext:t,$frontmatter:p}=x();return t.setup(),(v,l)=>{const i=o;return r(),c(f,m(_(e(k)(e(p),7))),{default:a(()=>[l[1]||(l[1]=s("h1",null,"一步求解流程",-1)),s("div",h,[u(i,d({},{title:"",ranges:[]}),{default:a(()=>[...l[0]||(l[0]=[s("pre",{class:"shiki shiki-themes vitesse-dark vitesse-light slidev-code",style:{"--shiki-dark":"#dbd7caee","--shiki-light":"#393a34","--shiki-dark-bg":"#121212","--shiki-light-bg":"#ffffff"}},[s("code",{class:"language-text"},[s("span",{class:"line"},[s("span",null,"swe_step():")]),n(`
`),s("span",{class:"line"},[s("span",null,"    ┌─────────────────────────────────┐")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │ 1. apply_bc()                    │  开放/入流边界条件")]),n(`
`),s("span",{class:"line"},[s("span",null,"    ├─────────────────────────────────┤")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │ 2. muscl_reconstruct_x()         │  x方向 MUSCL 重构")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │    muscl_reconstruct_y()         │  y方向 MUSCL 重构")]),n(`
`),s("span",{class:"line"},[s("span",null,"    ├─────────────────────────────────┤")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │ 3. hllc_flux_x()                 │  x界面 HLLC 通量")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │    hllc_flux_y()                 │  y界面 HLLC 通量")]),n(`
`),s("span",{class:"line"},[s("span",null,"    ├─────────────────────────────────┤")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │ 4. compute_dt()  ← CFL           │  自适应时间步")]),n(`
`),s("span",{class:"line"},[s("span",null,"    ├─────────────────────────────────┤")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │ 5. update_conserved()            │  更新 h, hu, hv")]),n(`
`),s("span",{class:"line"},[s("span",null,"    │    + 地形坡度源项                  │  + Manning 摩擦")]),n(`
`),s("span",{class:"line"},[s("span",null,"    └─────────────────────────────────┘")])])],-1)])]),_:1},16)]),l[2]||(l[2]=s("div",{class:"mt-4 text-gray-400 text-sm"},[s("p",null,[n("所有 kernel 均为 "),s("code",null,"@ti.kernel"),n("，在 GPU 上并行执行。每帧可能需要多个子步（substep）来推进到下一帧时间。")])],-1))]),_:1},16)}}};export{M as default};
