<template>
  <Transition name="fade">
    <div v-if="open" class="modal-backdrop code-overlay" @click.self="$emit('close')">
      <div class="code-modal" :class="{ 'has-tree': files.length > 0 }">
        <div class="modal-bar">
          <span class="modal-title">{{ filename }}</span>
          <button class="modal-close" @click="$emit('close')">&#10005;</button>
        </div>
        <!-- Mobile file tabs (shown < 780px when multi-file) -->
        <div v-if="files.length > 0" class="code-tabs">
          <span v-for="f in files" :key="f"
                class="code-tab" :class="{ active: f === activeFile }"
                @click="selectFile(f)">{{ f.split('/').pop() }}</span>
        </div>
        <div class="code-modal-content">
          <!-- Desktop file tree sidebar -->
          <div v-if="files.length > 0" class="code-tree">
            <template v-for="group in fileTree" :key="group.name">
              <template v-if="group.isFolder">
                <div class="code-tree-folder" @click="group.open = !group.open">
                  <span class="code-tree-arrow" :class="{ collapsed: !group.open }">&#9662;</span>
                  {{ group.name }}/
                </div>
                <template v-if="group.open">
                  <div v-for="f in group.children" :key="f"
                       class="code-tree-item" :class="{ active: f === activeFile }"
                       @click="selectFile(f)">{{ f.split('/').pop() }}</div>
                </template>
              </template>
              <div v-else class="code-tree-item root-file"
                   :class="{ active: group.name === activeFile }"
                   @click="selectFile(group.name)">{{ group.name }}</div>
            </template>
          </div>
          <!-- Code pane -->
          <div class="code-pane">
            <div class="code-modal-body">
              <pre><code v-html="html"></code></pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, computed, reactive, watch } from 'vue'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import javascript from 'highlight.js/lib/languages/javascript'
import typescript from 'highlight.js/lib/languages/typescript'
import glsl from 'highlight.js/lib/languages/glsl'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import json from 'highlight.js/lib/languages/json'
import xml from 'highlight.js/lib/languages/xml'
import 'highlight.js/styles/atom-one-dark.css'

hljs.registerLanguage('python', python)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('glsl', glsl)
hljs.registerLanguage('c', c)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('json', json)
hljs.registerLanguage('xml', xml)

const props = defineProps({
  open: { type: Boolean, default: false },
  post: { type: Object, default: null },
})
defineEmits(['close'])

const filename = ref('')
const html = ref('')
const files = ref([])
const activeFile = ref('')
const basePath = ref('')
const _folderOpen = reactive({})

const EXT_LANG = {
  py: 'python', js: 'javascript', ts: 'typescript',
  vert: 'glsl', frag: 'glsl', glsl: 'glsl',
  c: 'c', cpp: 'cpp', h: 'cpp',
  json: 'json', xml: 'xml', html: 'xml',
}

function langFromExt(fname) {
  const ext = fname.split('.').pop().toLowerCase()
  return EXT_LANG[ext] || 'plaintext'
}

async function selectFile(relPath) {
  activeFile.value = relPath
  filename.value = relPath
  html.value = ''
  const url = basePath.value + '/' + relPath
  const text = await fetch(url).then(r => r.text())
  const lang = langFromExt(relPath)
  html.value = hljs.highlight(text, { language: lang }).value
}

const fileTree = computed(() => {
  if (!files.value.length) return []
  const folders = {}
  const rootFiles = []
  for (const f of files.value) {
    const parts = f.split('/')
    if (parts.length > 1) {
      const dir = parts.slice(0, -1).join('/')
      if (!folders[dir]) folders[dir] = []
      folders[dir].push(f)
    } else {
      rootFiles.push(f)
    }
  }
  const result = []
  for (const f of rootFiles) {
    result.push({ isFolder: false, name: f })
  }
  for (const [dir, children] of Object.entries(folders)) {
    result.push({
      isFolder: true,
      name: dir,
      children,
      get open() { return _folderOpen[dir] !== false },
      set open(v) { _folderOpen[dir] = v },
    })
  }
  return result
})

watch(() => props.open, async (isOpen) => {
  if (!isOpen || !props.post) return
  const post = props.post
  html.value = ''

  if (post.codeFiles && post.codeFiles.length) {
    basePath.value = post.code
    files.value = post.codeFiles
    Object.keys(_folderOpen).forEach(k => delete _folderOpen[k])
    await selectFile(post.codeFiles[0])
  } else {
    basePath.value = ''
    files.value = []
    activeFile.value = ''
    filename.value = post.code.split('/').pop()
    const text = await fetch(post.code).then(r => r.text())
    const lang = langFromExt(post.code)
    html.value = hljs.highlight(text, { language: lang }).value
  }
})
</script>

<style scoped>
.modal-backdrop {
  position: fixed; inset: 0; z-index: 5000;
  background: rgba(0,0,0,.65); backdrop-filter: blur(4px);
  display: flex; justify-content: center; align-items: center;
}
.code-overlay { background: rgba(0,0,0,.75); }
.code-modal {
  width: 88vw; height: 85vh;
  border-radius: 10px; overflow: hidden;
  background: #1a1e2e;
  border: 1px solid rgba(167,139,250,.15);
  box-shadow: 0 8px 40px rgba(0,0,0,.5);
  display: flex; flex-direction: column;
}
.code-modal .modal-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: .4rem .7rem;
  background: rgba(167,139,250,.04);
  border-bottom: 1px solid rgba(167,139,250,.12);
}
.modal-title {
  font-family: var(--ff-mono); font-size: .55rem; color: var(--text-dim);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.modal-close {
  background: none; border: none; color: var(--text-dim);
  cursor: pointer; font-size: .8rem; padding: .1rem .35rem;
  border-radius: 4px; transition: background .15s, color .15s;
  display: flex; align-items: center; justify-content: center;
}
.modal-close:hover { background: rgba(255,255,255,.08); color: #fff; }
.code-modal-content {
  flex: 1; display: flex; overflow: hidden;
}
/* ── File tree sidebar ── */
.code-tree {
  width: 180px; min-width: 140px;
  background: #141824;
  border-right: 1px solid rgba(167,139,250,.12);
  overflow-y: auto; padding: .35rem 0;
  font-family: var(--ff-mono); font-size: .55rem;
}
.code-tree-folder {
  padding: .25rem .5rem; color: var(--text-dim); cursor: pointer;
  display: flex; align-items: center; gap: .25rem;
  user-select: none;
}
.code-tree-folder:hover { background: rgba(167,139,250,.06); }
.code-tree-arrow { display: inline-block; transition: transform .15s; font-size: .5rem; }
.code-tree-arrow.collapsed { transform: rotate(-90deg); }
.code-tree-item {
  padding: .22rem .5rem .22rem 1.3rem; color: var(--text-dim);
  cursor: pointer; transition: background .1s, color .1s;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.code-tree-item.root-file { padding-left: .5rem; }
.code-tree-item:hover { background: rgba(167,139,250,.08); color: rgba(255,255,255,.8); }
.code-tree-item.active { background: rgba(167,139,250,.14); color: #a78bfa; }
/* ── Code pane ── */
.code-pane {
  flex: 1; overflow: auto; min-width: 0;
}
.code-modal-body {
  padding: .5rem .7rem; overflow: auto; height: 100%;
}
.code-modal-body pre {
  margin: 0; background: transparent !important;
  font-family: var(--ff-mono); font-size: .65rem; line-height: 1.55;
  white-space: pre; overflow-x: auto;
}
.code-modal-body code { background: none !important; padding: 0 !important; }

/* ── Mobile file tabs ── */
.code-tabs {
  display: none; overflow-x: auto; white-space: nowrap; background: #141824;
  border-bottom: 1px solid rgba(167,139,250,.12); padding: .25rem .5rem;
  scrollbar-width: none; gap: .2rem;
}
.code-tabs::-webkit-scrollbar { display: none; }
.code-tab {
  display: inline-block; padding: .3rem .7rem; font-family: var(--ff-mono); font-size: .65rem;
  color: rgba(255,255,255,.5); border-radius: 4px; cursor: pointer;
  transition: background .1s, color .1s; flex-shrink: 0;
}
.code-tab:hover { background: rgba(167,139,250,.1); color: rgba(255,255,255,.8); }
.code-tab.active { background: rgba(167,139,250,.18); color: #a78bfa; }

@media (max-width: 780px) {
  .code-tree { display: none; }
  .code-tabs { display: flex; }
}
</style>
