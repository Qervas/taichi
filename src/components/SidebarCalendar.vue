<template>
  <div class="panel">
    <div class="panel-hdr">
      <span>// Date Filter</span>
    </div>
    <div class="cal-wrap">
      <div class="cal-nav">
        <button class="cal-btn" @click="prevMonth">&#8249;</button>
        <span class="cal-month-lbl">{{ calMonthLabel }}</span>
        <button class="cal-btn" @click="nextMonth">&#8250;</button>
      </div>
      <div class="cal-grid">
        <div v-for="d in DOW" :key="d" class="cal-dow">{{ d }}</div>
        <template v-for="(slot, i) in calDays" :key="i">
          <div v-if="slot === null" class="cal-day"></div>
          <div v-else
            class="cal-day"
            :class="{
              'has-posts': slot.hasPosts,
              'today':     slot.isToday,
              'active':    slot.dateStr === activeDate
            }"
            @click="slot.hasPosts && toggleDate(slot.dateStr)"
          >{{ slot.day }}</div>
        </template>
      </div>
      <button v-if="activeDate" class="cal-clear" @click="$emit('update:activeDate', null)">&#10005; &nbsp;Clear date filter</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { DOW, MONTHS, POST_DATES, todayStr } from '../data/posts.js'

const props = defineProps({
  activeDate: { type: String, default: null },
})
const emit = defineEmits(['update:activeDate'])

const _now = new Date()
const calYear = ref(_now.getFullYear())
const calMonth = ref(_now.getMonth())

const calMonthLabel = computed(() => `${MONTHS[calMonth.value]} ${calYear.value}`)

const calDays = computed(() => {
  const days = []
  const firstDow = new Date(calYear.value, calMonth.value, 1).getDay()
  const daysInMo = new Date(calYear.value, calMonth.value + 1, 0).getDate()

  for (let i = 0; i < firstDow; i++) days.push(null)

  for (let d = 1; d <= daysInMo; d++) {
    const ds = `${calYear.value}-${String(calMonth.value + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`
    days.push({ day: d, dateStr: ds, hasPosts: POST_DATES.has(ds), isToday: ds === todayStr })
  }
  return days
})

function prevMonth() {
  calMonth.value--
  if (calMonth.value < 0) { calMonth.value = 11; calYear.value-- }
}
function nextMonth() {
  calMonth.value++
  if (calMonth.value > 11) { calMonth.value = 0; calYear.value++ }
}
function toggleDate(ds) {
  emit('update:activeDate', props.activeDate === ds ? null : ds)
}
</script>

<style scoped>
.panel {
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-card); overflow: hidden;
}
.panel-hdr {
  font-family: var(--ff-mono); font-size: .55rem;
  color: var(--cyan-dim); padding: .4rem .65rem;
  border-bottom: 1px solid var(--border);
  letter-spacing: .04em;
}
.cal-wrap { padding: .45rem .55rem .55rem; }
.cal-nav {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: .3rem;
}
.cal-month-lbl {
  font-family: var(--ff-mono); font-size: .6rem; color: var(--text-dim);
}
.cal-btn {
  background: transparent; border: none; color: var(--text-dim);
  font-size: 1rem; cursor: pointer; padding: 0 .25rem;
  transition: color .15s;
}
.cal-btn:hover { color: var(--cyan); }
.cal-grid {
  display: grid; grid-template-columns: repeat(7, 1fr);
  text-align: center; font-family: var(--ff-mono);
}
.cal-dow {
  font-size: .45rem; color: var(--text-dim);
  padding: .15rem 0; text-transform: uppercase;
}
.cal-day {
  font-size: .52rem; color: var(--text-dim);
  padding: .18rem 0; border-radius: 3px; cursor: default;
}
.cal-day.has-posts {
  color: var(--cyan); cursor: pointer;
  font-weight: 600;
}
.cal-day.has-posts:hover { background: rgba(0,212,255,.08); }
.cal-day.today {
  border: 1px solid var(--cyan-dim);
}
.cal-day.active {
  background: var(--cyan); color: #000;
  font-weight: 700;
}
.cal-clear {
  display: block; width: 100%; margin-top: .5rem;
  background: none; border: none; color: var(--cyan-dim);
  font-family: var(--ff-mono); font-size: .5rem;
  cursor: pointer; text-align: center;
  transition: color .15s;
}
.cal-clear:hover { color: var(--cyan); }
</style>
