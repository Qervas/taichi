<template>
  <Transition name="fade">
    <div v-if="open" class="modal-backdrop vid-overlay" @click.self="$emit('close')">
      <div class="vid-modal">
        <div class="modal-bar">
          <span class="modal-title">{{ title }}</span>
          <button class="modal-close" @click="$emit('close')">&#10005;</button>
        </div>
        <div class="vid-modal-body">
          <video controls autoplay playsinline :src="url"></video>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
defineProps({
  open: { type: Boolean, default: false },
  title: { type: String, default: '' },
  url: { type: String, default: '' },
})
defineEmits(['close'])
</script>

<style scoped>
.modal-backdrop {
  position: fixed; inset: 0; z-index: 5000;
  background: rgba(0,0,0,.65); backdrop-filter: blur(4px);
  display: flex; justify-content: center; align-items: center;
}
.vid-overlay { background: rgba(0,0,0,.8); }
.vid-modal {
  width: min(90vw, 960px);
  border-radius: 10px; overflow: hidden;
  background: #000; border: 1px solid var(--border);
  box-shadow: 0 8px 40px rgba(0,0,0,.6);
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
.modal-close {
  background: none; border: none; color: var(--text-dim);
  cursor: pointer; font-size: .8rem; padding: .1rem .35rem;
  border-radius: 4px; transition: background .15s, color .15s;
  display: flex; align-items: center; justify-content: center;
}
.modal-close:hover { background: rgba(255,255,255,.08); color: #fff; }
.vid-modal-body { position: relative; aspect-ratio: 16/9; }
.vid-modal-body video { width: 100%; height: 100%; display: block; }
</style>
