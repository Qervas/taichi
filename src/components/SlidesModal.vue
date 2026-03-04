<template>
  <Transition name="fade">
    <div v-if="open" class="modal-backdrop" @click.self="$emit('close')">
      <div class="modal" ref="modalEl">
        <div class="modal-bar">
          <span class="modal-title">{{ title }}</span>
          <div class="modal-actions">
            <button class="modal-fs" @click="toggleFullscreen" title="Fullscreen">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 00-2 2v3M16 3h3a2 2 0 012 2v3M8 21H5a2 2 0 01-2-2v-3M16 21h3a2 2 0 002-2v-3"/></svg>
            </button>
            <button class="modal-close" @click="$emit('close')">&#10005;</button>
          </div>
        </div>
        <div class="modal-body">
          <div v-if="loading" class="slides-loader">
            <div class="slides-loader-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <rect x="2" y="3" width="20" height="14" rx="2"/>
                <path d="M8 21h8M12 17v4"/>
              </svg>
            </div>
            <div class="slides-loader-bar"><div class="slides-loader-fill"></div></div>
            <span class="slides-loader-text">Loading slides…</span>
          </div>
          <iframe
            :src="url"
            title="Slide viewer"
            allowfullscreen
            @load="onIframeLoad"
          ></iframe>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  open: { type: Boolean, default: false },
  title: { type: String, default: '' },
  url: { type: String, default: '' },
})
defineEmits(['close'])

const loading = ref(false)
let openedAt = 0

watch(() => props.open, (isOpen) => {
  if (isOpen) {
    loading.value = true
    openedAt = Date.now()
  }
})

function onIframeLoad() {
  // Show loader for at least 600ms so it's visible
  const elapsed = Date.now() - openedAt
  const delay = Math.max(0, 600 - elapsed)
  setTimeout(() => { loading.value = false }, delay)
}

const modalEl = ref(null)

async function toggleFullscreen() {
  if (!document.fullscreenElement) {
    modalEl.value?.requestFullscreen()
  } else {
    document.exitFullscreen()
  }
}
</script>

<style scoped>
.modal-backdrop {
  position: fixed; inset: 0; z-index: 5000;
  background: rgba(0,0,0,.65); backdrop-filter: blur(4px);
  display: flex; justify-content: center; align-items: center;
}
.modal {
  width: 88vw; height: 85vh;
  border-radius: 10px; overflow: hidden;
  background: #0a0f1f;
  border: 1px solid var(--border);
  box-shadow: 0 8px 40px rgba(0,0,0,.5);
  display: flex; flex-direction: column;
}
.modal-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: .4rem .7rem;
  background: rgba(255,255,255,.03);
  border-bottom: 1px solid var(--border);
}
.modal-title {
  font-family: var(--ff-mono); font-size: .55rem; color: var(--text-dim);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.modal-actions { display: flex; gap: .25rem; }
.modal-close, .modal-fs {
  background: none; border: none; color: var(--text-dim);
  cursor: pointer; font-size: .8rem; padding: .1rem .35rem;
  border-radius: 4px; transition: background .15s, color .15s;
  display: flex; align-items: center; justify-content: center;
}
.modal-close:hover, .modal-fs:hover { background: rgba(255,255,255,.08); color: #fff; }
.modal-body { flex: 1; position: relative; }
.modal-body iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: none; }

.slides-loader {
  position: absolute; inset: 0; z-index: 1;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  gap: .8rem;
  background: #0a0f1f;
}
.slides-loader-icon {
  width: 36px; height: 36px; color: var(--cyan-dim);
  animation: loader-breathe 2s ease-in-out infinite;
}
.slides-loader-icon svg { width: 100%; height: 100%; }
@keyframes loader-breathe { 0%,100%{opacity:.4;transform:scale(.95)} 50%{opacity:1;transform:scale(1.05)} }
.slides-loader-bar {
  width: 220px; height: 3px;
  background: rgba(255,255,255,.08); border-radius: 3px; overflow: hidden;
}
.slides-loader-fill {
  height: 100%; width: 30%;
  background: linear-gradient(90deg, var(--cyan), var(--purple));
  border-radius: 3px;
  animation: slides-pulse 1.6s ease-in-out infinite;
}
@keyframes slides-pulse {
  0%   { width: 10%; margin-left: 0; }
  50%  { width: 50%; margin-left: 25%; }
  100% { width: 10%; margin-left: 90%; }
}
.slides-loader-text {
  font-family: var(--ff-mono); font-size: .6rem; color: var(--text-dim);
  letter-spacing: .06em;
}

@media (max-width: 780px) {
  .modal { width: 98vw; height: 92vh; }
}
</style>
