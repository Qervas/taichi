<template>
  <div class="explorer">
    <!-- Mobile project selector -->
    <div class="mobile-project-select">
      <select :value="activeProject?.id" @change="selectProjectById($event.target.value)">
        <option v-for="p in projects" :key="p.id" :value="p.id">{{ p.phase }} — {{ p.title }}</option>
      </select>
    </div>

    <div class="explorer-body">
      <!-- Panel 1: Project list -->
      <div class="panel-projects">
        <div class="panel-header">EXPLORER</div>
        <div
          v-for="p in projects" :key="p.id"
          class="project-item" :class="{ active: activeProject?.id === p.id }"
          @click="selectProject(p)"
        >
          <div class="project-phase">{{ p.phase }}</div>
          <div class="project-title">{{ p.title }}</div>
          <span class="project-badge">{{ p.codeFiles ? p.codeFiles.length : 1 }}</span>
        </div>
      </div>

      <!-- Panel 2: File tree -->
      <div class="panel-tree">
        <div class="panel-header">{{ activeProject?.phase || 'FILES' }}</div>
        <template v-if="activeProject">
          <!-- Multi-file tree -->
          <template v-if="fileTree.length">
            <template v-for="group in fileTree" :key="group.name">
              <template v-if="group.isFolder">
                <div class="tree-folder" @click="group.open = !group.open">
                  <span class="tree-arrow" :class="{ collapsed: !group.open }">&#9662;</span>
                  <span class="tree-folder-icon">&#128193;</span>
                  {{ group.name }}
                </div>
                <template v-if="group.open">
                  <div v-for="f in group.children" :key="f"
                       class="tree-file tree-file--nested"
                       :class="{ active: isFileActive(f) }"
                       @click="openFile(f)">
                    <span class="file-icon" :style="{ color: langColor(f) }">&#9679;</span>
                    {{ f.split('/').pop() }}
                  </div>
                </template>
              </template>
              <div v-else class="tree-file"
                   :class="{ active: isFileActive(group.name) }"
                   @click="openFile(group.name)">
                <span class="file-icon" :style="{ color: langColor(group.name) }">&#9679;</span>
                {{ group.name }}
              </div>
            </template>
          </template>
          <!-- Single-file -->
          <div v-else class="tree-file active">
            <span class="file-icon" :style="{ color: langColor(singleFileName) }">&#9679;</span>
            {{ singleFileName }}
          </div>
        </template>
      </div>

      <!-- Panel 3: Editor area -->
      <div class="panel-editor">
        <!-- Tab bar -->
        <div class="tab-bar">
          <div class="tab-list">
            <div v-for="tab in openTabs" :key="tab.key"
                 class="tab" :class="{ active: tab.key === activeTabKey }"
                 @click="switchTab(tab)">
              <span class="tab-dot" :style="{ color: langColor(tab.name) }">&#9679;</span>
              <span class="tab-name">{{ tab.name.split('/').pop() }}</span>
              <span class="tab-close" @click.stop="closeTab(tab)">&#10005;</span>
            </div>
          </div>
          <div class="tab-actions">
            <div class="font-size-ctrl">
              <button class="fs-btn" @click="changeFontSize(-1)" title="Decrease font size">A-</button>
              <span class="fs-label">{{ fontSize }}px</span>
              <button class="fs-btn" @click="changeFontSize(1)" title="Increase font size">A+</button>
            </div>
            <button v-if="activeProject" class="share-btn" @click="shareProject" :title="'Copy link to ' + activeProject.phase">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
              {{ copied ? 'Copied!' : 'Share' }}
            </button>
            <button v-if="activeProject && canDownload" class="dl-btn" :disabled="zipping" @click="downloadZip">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              {{ zipping ? 'Zipping...' : 'Download' }}
            </button>
          </div>
        </div>

        <!-- Breadcrumb -->
        <div v-if="activeTab" class="breadcrumb">
          <span class="bc-seg">{{ activeProject?.phase }}</span>
          <template v-for="(seg, i) in activeTab.name.split('/')" :key="i">
            <span class="bc-sep">/</span>
            <span class="bc-seg" :class="{ 'bc-file': i === activeTab.name.split('/').length - 1 }">{{ seg }}</span>
          </template>
        </div>

        <!-- Code pane -->
        <div class="code-area" v-if="activeTab" :style="{ fontSize: fontSize + 'px' }">
          <div class="line-numbers" ref="lineNumsRef">
            <span v-for="n in lineCount" :key="n">{{ n }}</span>
          </div>
          <div class="code-content" @scroll="syncScroll">
            <pre><code v-html="activeTab.html"></code></pre>
          </div>
        </div>

        <!-- Empty state -->
        <div v-else class="empty-editor">
          <div class="empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity=".3">
              <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
            </svg>
          </div>
          <div class="empty-text">Select a file to view</div>
          <div class="empty-hint">Browse projects on the left, then pick a file</div>
        </div>
      </div>
    </div>

    <!-- Mobile file tabs (shown < 780px when multi-file) -->
    <div v-if="activeProject && fileTree.length" class="mobile-tabs">
      <span v-for="f in activeProject.codeFiles" :key="f"
            class="m-tab" :class="{ active: isFileActive(f) }"
            @click="openFile(f)">{{ f.split('/').pop() }}</span>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, watch, nextTick } from 'vue'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'
import javascript from 'highlight.js/lib/languages/javascript'
import typescript from 'highlight.js/lib/languages/typescript'
import glsl from 'highlight.js/lib/languages/glsl'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import json from 'highlight.js/lib/languages/json'
import xml from 'highlight.js/lib/languages/xml'
import bash from 'highlight.js/lib/languages/bash'
import makefile from 'highlight.js/lib/languages/makefile'
import 'highlight.js/styles/atom-one-dark.css'

hljs.registerLanguage('python', python)
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('typescript', typescript)
hljs.registerLanguage('glsl', glsl)
hljs.registerLanguage('c', c)
hljs.registerLanguage('cpp', cpp)
hljs.registerLanguage('json', json)
hljs.registerLanguage('xml', xml)
hljs.registerLanguage('bash', bash)
hljs.registerLanguage('makefile', makefile)

const props = defineProps({
  projects: { type: Array, required: true },
  initialProject: { type: String, default: null },
})

const EXT_LANG = {
  py: 'python', js: 'javascript', ts: 'typescript',
  vert: 'glsl', frag: 'glsl', glsl: 'glsl',
  c: 'c', cpp: 'cpp', h: 'cpp', cu: 'cpp',
  json: 'json', xml: 'xml', html: 'xml',
  sh: 'bash', Makefile: 'makefile',
}

const LANG_COLORS = {
  python: '#3572A5', javascript: '#f1e05a', typescript: '#3178c6',
  glsl: '#5686a5', c: '#555555', cpp: '#f34b7d',
  json: '#a6e22e', xml: '#e44b23', bash: '#89e051', makefile: '#427819',
  plaintext: '#888',
}

function langFromExt(fname) {
  const base = fname.split('/').pop()
  if (base === 'Makefile') return 'makefile'
  const ext = base.split('.').pop().toLowerCase()
  return EXT_LANG[ext] || 'plaintext'
}

function langColor(fname) {
  return LANG_COLORS[langFromExt(fname)] || LANG_COLORS.plaintext
}

// State
const activeProject = ref(null)
const openTabs = ref([])
const activeTabKey = ref(null)
const _folderOpen = reactive({})
const zipping = ref(false)
const copied = ref(false)
let _copiedTimer = null

function shareProject() {
  if (!activeProject.value) return
  const url = `${location.origin}${location.pathname}#code/${activeProject.value.id}`
  navigator.clipboard.writeText(url).then(() => {
    clearTimeout(_copiedTimer)
    copied.value = true
    _copiedTimer = setTimeout(() => { copied.value = false }, 2000)
  })
}
const lineNumsRef = ref(null)
const fontSize = ref(parseInt(localStorage.getItem('code-font-size')) || 13)

function changeFontSize(delta) {
  fontSize.value = Math.min(24, Math.max(10, fontSize.value + delta))
  localStorage.setItem('code-font-size', fontSize.value)
}

const activeTab = computed(() => openTabs.value.find(t => t.key === activeTabKey.value) || null)
const lineCount = computed(() => {
  if (!activeTab.value?.html) return 0
  return (activeTab.value.html.match(/\n/g) || []).length + 1
})

const singleFileName = computed(() => {
  if (!activeProject.value || activeProject.value.codeFiles) return ''
  return activeProject.value.code.split('/').pop()
})

const canDownload = computed(() => {
  return activeProject.value?.codeFiles && activeProject.value.codeFiles.length > 0
})

const fileTree = computed(() => {
  if (!activeProject.value?.codeFiles) return []
  const folders = {}
  const rootFiles = []
  for (const f of activeProject.value.codeFiles) {
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
  for (const f of rootFiles) result.push({ isFolder: false, name: f })
  for (const [dir, children] of Object.entries(folders)) {
    result.push({
      isFolder: true, name: dir, children,
      get open() { return _folderOpen[dir] !== false },
      set open(v) { _folderOpen[dir] = v },
    })
  }
  return result
})

function isFileActive(name) {
  return activeTabKey.value === (activeProject.value?.id + ':' + name)
}

function selectProject(p) {
  if (activeProject.value?.id === p.id) return
  activeProject.value = p
  Object.keys(_folderOpen).forEach(k => delete _folderOpen[k])
  // Auto-open the first file
  if (p.codeFiles && p.codeFiles.length) {
    openFile(p.codeFiles[0])
  } else {
    openSingleFile(p)
  }
  // Update hash
  history.replaceState(null, '', '#code/' + p.id)
}

function selectProjectById(id) {
  const p = props.projects.find(pr => pr.id === id)
  if (p) selectProject(p)
}

async function openFile(relPath) {
  const key = activeProject.value.id + ':' + relPath
  const existing = openTabs.value.find(t => t.key === key)
  if (existing) {
    activeTabKey.value = key
    return
  }
  // Fetch and highlight
  const url = activeProject.value.code + '/' + relPath
  const text = await fetch(url).then(r => r.text())
  const lang = langFromExt(relPath)
  const highlighted = hljs.highlight(text, { language: lang }).value
  const tab = { key, name: relPath, html: highlighted, projectId: activeProject.value.id }
  openTabs.value.push(tab)
  activeTabKey.value = key
}

async function openSingleFile(p) {
  const key = p.id + ':' + p.code
  const existing = openTabs.value.find(t => t.key === key)
  if (existing) {
    activeTabKey.value = key
    return
  }
  const text = await fetch(p.code).then(r => r.text())
  const lang = langFromExt(p.code)
  const highlighted = hljs.highlight(text, { language: lang }).value
  const tab = { key, name: p.code.split('/').pop(), html: highlighted, projectId: p.id }
  openTabs.value.push(tab)
  activeTabKey.value = key
}

function switchTab(tab) {
  activeTabKey.value = tab.key
}

function closeTab(tab) {
  const idx = openTabs.value.findIndex(t => t.key === tab.key)
  openTabs.value.splice(idx, 1)
  if (activeTabKey.value === tab.key) {
    if (openTabs.value.length) {
      activeTabKey.value = openTabs.value[Math.min(idx, openTabs.value.length - 1)].key
    } else {
      activeTabKey.value = null
    }
  }
}

function syncScroll(e) {
  if (lineNumsRef.value) {
    lineNumsRef.value.scrollTop = e.target.scrollTop
  }
}

async function downloadZip() {
  if (!activeProject.value?.codeFiles || zipping.value) return
  zipping.value = true
  try {
    const { default: JSZip } = await import('jszip')
    const zip = new JSZip()
    for (const file of activeProject.value.codeFiles) {
      const url = `${activeProject.value.code}/${file}`
      const resp = await fetch(url)
      const text = await resp.text()
      zip.file(file, text)
    }
    const blob = await zip.generateAsync({ type: 'blob' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${activeProject.value.phase} - ${activeProject.value.title}.zip`
    a.click()
    URL.revokeObjectURL(a.href)
  } finally {
    zipping.value = false
  }
}

// Init: select initial project if provided
watch(() => props.initialProject, (id) => {
  if (id) {
    const p = props.projects.find(pr => pr.id === id)
    if (p) { selectProject(p); return }
  }
  if (!activeProject.value && props.projects.length) {
    selectProject(props.projects[0])
  }
}, { immediate: true })
</script>

<style scoped>
.explorer {
  display: flex; flex-direction: column;
  height: calc(100vh - 46px);
  background: var(--bg);
}
.explorer-body {
  flex: 1; display: flex; overflow: hidden;
}

/* ── Panel headers ── */
.panel-header {
  font-family: var(--ff-mono); font-size: .5rem;
  color: var(--text-dim); letter-spacing: .12em;
  padding: .45rem .6rem; text-transform: uppercase;
  border-bottom: 1px solid rgba(167,139,250,.08);
  user-select: none;
}

/* ── Panel 1: Projects ── */
.panel-projects {
  width: 180px; min-width: 140px;
  background: #0c1020;
  border-right: 1px solid rgba(167,139,250,.08);
  overflow-y: auto;
  flex-shrink: 0;
}
.project-item {
  padding: .4rem .6rem; cursor: pointer;
  border-left: 2px solid transparent;
  transition: background .12s, border-color .12s;
  position: relative;
}
.project-item:hover { background: rgba(167,139,250,.04); }
.project-item.active {
  background: rgba(167,139,250,.08);
  border-left-color: var(--purple);
}
.project-phase {
  font-family: var(--ff-brand); font-size: .4rem;
  color: var(--cyan-dim); letter-spacing: .08em;
  text-transform: uppercase;
}
.project-title {
  font-family: var(--ff-mono); font-size: .5rem;
  color: var(--text); line-height: 1.4;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.project-badge {
  position: absolute; right: .5rem; top: 50%; transform: translateY(-50%);
  font-family: var(--ff-mono); font-size: .42rem;
  background: rgba(167,139,250,.12); color: var(--purple);
  padding: .08rem .3rem; border-radius: 8px;
}

/* ── Panel 2: File tree ── */
.panel-tree {
  width: 200px; min-width: 160px;
  background: #0e1324;
  border-right: 1px solid rgba(167,139,250,.08);
  overflow-y: auto;
  flex-shrink: 0;
}
.tree-folder {
  display: flex; align-items: center; gap: .25rem;
  padding: .25rem .5rem;
  font-family: var(--ff-mono); font-size: .52rem;
  color: var(--text-dim); cursor: pointer; user-select: none;
}
.tree-folder:hover { background: rgba(167,139,250,.05); }
.tree-arrow {
  display: inline-block; font-size: .45rem;
  transition: transform .15s;
}
.tree-arrow.collapsed { transform: rotate(-90deg); }
.tree-folder-icon { font-size: .55rem; }
.tree-file {
  display: flex; align-items: center; gap: .3rem;
  padding: .2rem .5rem .2rem .6rem;
  font-family: var(--ff-mono); font-size: .52rem;
  color: var(--text-dim); cursor: pointer;
  transition: background .1s, color .1s;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.tree-file--nested { padding-left: 1.4rem; }
.tree-file:hover { background: rgba(167,139,250,.06); color: rgba(255,255,255,.8); }
.tree-file.active { background: rgba(167,139,250,.12); color: #a78bfa; }
.file-icon { font-size: .4rem; flex-shrink: 0; }

/* ── Panel 3: Editor ── */
.panel-editor {
  flex: 1; display: flex; flex-direction: column;
  min-width: 0; background: #111628;
}

/* Tab bar */
.tab-bar {
  display: flex; align-items: center;
  background: #0c1020;
  border-bottom: 1px solid rgba(167,139,250,.08);
  min-height: 32px;
}
.tab-list {
  display: flex; overflow-x: auto; flex: 1;
  scrollbar-width: none;
}
.tab-list::-webkit-scrollbar { display: none; }
.tab {
  display: flex; align-items: center; gap: .25rem;
  padding: .3rem .6rem;
  font-family: var(--ff-mono); font-size: .5rem;
  color: var(--text-dim); cursor: pointer;
  border-right: 1px solid rgba(167,139,250,.06);
  white-space: nowrap; flex-shrink: 0;
  transition: background .1s, color .1s;
}
.tab:hover { background: rgba(167,139,250,.06); }
.tab.active {
  background: #111628;
  color: #fff;
  border-bottom: 2px solid var(--purple);
}
.tab-dot { font-size: .35rem; }
.tab-close {
  font-size: .5rem; opacity: 0; padding: 0 .15rem;
  border-radius: 3px; transition: opacity .1s, background .1s;
  cursor: pointer;
}
.tab:hover .tab-close { opacity: .5; }
.tab-close:hover { opacity: 1 !important; background: rgba(255,255,255,.1); }
.tab-actions {
  padding: 0 .5rem; flex-shrink: 0;
}
.font-size-ctrl {
  display: inline-flex; align-items: center; gap: .15rem;
  margin-right: .4rem;
}
.fs-btn {
  font-family: var(--ff-mono); font-size: .48rem;
  padding: .15rem .35rem; border-radius: 3px;
  background: rgba(255,255,255,.05); color: var(--text-dim);
  border: 1px solid rgba(255,255,255,.08);
  cursor: pointer; transition: background .12s, color .12s;
}
.fs-btn:hover { background: rgba(255,255,255,.1); color: #fff; }
.fs-label {
  font-family: var(--ff-mono); font-size: .45rem;
  color: var(--text-dim); min-width: 2.2rem; text-align: center;
}
.share-btn {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: var(--ff-mono); font-size: .48rem;
  padding: .2rem .5rem; border-radius: 4px;
  background: rgba(167,139,250,.08); color: #a78bfa;
  border: 1px solid rgba(167,139,250,.18);
  cursor: pointer; transition: background .15s;
  margin-right: .3rem;
}
.share-btn:hover { background: rgba(167,139,250,.16); }
.dl-btn {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: var(--ff-mono); font-size: .48rem;
  padding: .2rem .5rem; border-radius: 4px;
  background: rgba(52,211,153,.08); color: #34d399;
  border: 1px solid rgba(52,211,153,.18);
  cursor: pointer; transition: background .15s;
}
.dl-btn:hover { background: rgba(52,211,153,.16); }
.dl-btn:disabled { opacity: .5; cursor: wait; }

/* Breadcrumb */
.breadcrumb {
  display: flex; align-items: center; gap: .15rem;
  padding: .25rem .7rem;
  font-family: var(--ff-mono); font-size: .48rem;
  color: var(--text-dim);
  background: rgba(167,139,250,.02);
  border-bottom: 1px solid rgba(167,139,250,.05);
}
.bc-sep { opacity: .3; }
.bc-file { color: #fff; }

/* Code area */
.code-area {
  flex: 1; display: flex; overflow: hidden;
  position: relative;
}
.line-numbers {
  width: 48px; flex-shrink: 0;
  overflow: hidden;
  padding: .5rem 0;
  text-align: right;
  font-family: var(--ff-mono); font-size: inherit;
  line-height: 1.55;
  color: rgba(255,255,255,.18);
  user-select: none;
  background: rgba(0,0,0,.15);
  border-right: 1px solid rgba(167,139,250,.06);
}
.line-numbers span {
  display: block; padding-right: .5rem;
}
.code-content {
  flex: 1; overflow: auto; padding: .5rem .7rem;
}
.code-content pre {
  margin: 0; background: transparent !important;
  font-family: var(--ff-mono); font-size: inherit;
  line-height: 1.55; white-space: pre; overflow-x: visible;
}
.code-content code {
  background: none !important; padding: 0 !important;
}

/* Empty state */
.empty-editor {
  flex: 1; display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  gap: .4rem;
}
.empty-text {
  font-family: var(--ff-mono); font-size: .6rem;
  color: var(--text-dim);
}
.empty-hint {
  font-family: var(--ff-mono); font-size: .48rem;
  color: rgba(255,255,255,.25);
}

/* ── Mobile ── */
.mobile-project-select {
  display: none;
  padding: .4rem .6rem;
  background: #0c1020;
  border-bottom: 1px solid rgba(167,139,250,.08);
}
.mobile-project-select select {
  width: 100%;
  font-family: var(--ff-mono); font-size: .55rem;
  padding: .35rem .5rem; border-radius: 6px;
  background: #111628; color: var(--text);
  border: 1px solid rgba(167,139,250,.15);
  outline: none;
}
.mobile-tabs {
  display: none;
  overflow-x: auto; white-space: nowrap;
  background: #0c1020;
  border-top: 1px solid rgba(167,139,250,.08);
  padding: .25rem .5rem;
  scrollbar-width: none; gap: .2rem;
}
.mobile-tabs::-webkit-scrollbar { display: none; }
.m-tab {
  display: inline-block; padding: .3rem .7rem;
  font-family: var(--ff-mono); font-size: .55rem;
  color: rgba(255,255,255,.5); border-radius: 4px;
  cursor: pointer; transition: background .1s, color .1s;
}
.m-tab:hover { background: rgba(167,139,250,.1); color: rgba(255,255,255,.8); }
.m-tab.active { background: rgba(167,139,250,.18); color: #a78bfa; }

@media (max-width: 780px) {
  .panel-projects, .panel-tree { display: none; }
  .mobile-project-select { display: block; }
  .mobile-tabs { display: flex; }
  .explorer { height: calc(100vh - 42px); }
}
</style>
