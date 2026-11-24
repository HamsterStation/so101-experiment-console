'use strict';
(() => {
  const section = document.querySelector('[data-view="trial"]');
  const here = () => !section.hidden && !document.hidden;
  const actions = ['check', 'start', 'stop', 'release'];
  const el = name => $('trial-' + name);
  let trial = {}, reachable = false, reading = false, pending = '', waitingId = '';
  let lastId = '', previousPhase = '', videoId = '';
  const requestedModel = new URLSearchParams(location.search).get('trial_model');
  if (Array.from(el('model').options).some(option => option.value === requestedModel)) {
    el('model').value = requestedModel;
    el('zone').value = 'A1';
  }
  const activeTrial = () => trial.status === 'running';

  const outcome = j => j.task_complete ? '方块已入盒，任务完成 · ' : '';

  function draw() {
    const busy = activeTrial(), phase = busy ? trial.phase : '';
    if (trial.id !== lastId || phase !== previousPhase && phase === 'ready') el('onsite').checked = false;
    lastId = trial.id;
    previousPhase = phase;
    if (busy) el('model').value = trial.model || 'cube49';
    el('model').disabled = busy || !!pending;
    const hasCenter = ['cube46-kd', 'cube46-kd-int8', 'smolvla46', 'smolvla46-4080', 'smolvla46-4080-direct'].includes(el('model').value);
    const fourZones = hasCenter || el('model').value === 'cube46';
    for (const option of el('zone').options) {
      option.disabled = option.hidden = (fourZones && ['A3','B3'].includes(option.value)) || (!hasCenter && option.value === '四点中心');
    }
    if (el('zone').selectedOptions[0]?.disabled) el('zone').value = 'A1';
    if (busy && trial.zone) el('zone').value = trial.zone;
    el('model-hint').textContent = fourZones
      ? '方块落入盒内后，点击「停止跟随」。本轮会保存双摄回放。'
      : '检测到夹爪松开、方块稳定留在盒内后自动停止跟随。';
    el('zone').disabled = busy || !!pending;
    const preparing = phase === 'ready' && trial.owned;
    el('onsite-label').hidden = !preparing;
    el('check').hidden = busy;
    el('start').hidden = !preparing;
    const stoppable = ['checking','ready','arming','running','stopping'].includes(phase);
    el('stop').hidden = !stoppable;
    el('stop').textContent = ['checking','ready'].includes(phase) ? '取消预检' : '停止跟随';
    el('release').hidden = !['holding','releasing'].includes(phase);
    el('check').disabled = !reachable || !token || !!pending || busy;
    el('start').disabled = !reachable || !preparing || !el('onsite').checked || !!pending;
    el('stop').disabled = !stoppable || phase === 'stopping' || pending === 'stop';
    el('release').disabled = !reachable || phase !== 'holding' || !!pending;
    const text = trial.detail || '准备就绪，可先检查双摄、从臂和模型首步。';
    el('status').textContent = reachable ? outcome(trial) + text +
      (phase==='ready' && !trial.owned ? ' · 请在发起预检的标签页确认接管。' : '') +
      (busy && trial.fault ? ' · ' + trial.fault : '') :
      (trial.armed_possible && busy ? '控制连接中断。请在现场观察从臂，必要时托稳后断电。' : '正在连接本机实验台…');
    const history = arr(state.jobs).filter(j=>j.kind==='trial').slice(0,8);
    const reference = state.trial_reference ? '<div class="row"><div><strong>上一次入盒回放</strong><p>ACT 4,000 步 · 现场已确认方块落盒，起点未标记。</p></div><button type="button" class="soft" data-trial-video="reference">双摄回放</button></div>' : '';
    markup('trial-history',history.map(j=>'<div class="row"><div><strong>'+esc(j.name)+'</strong><p>'+
      esc((j.created||'').replace('T',' ').slice(5,19)+' · '+outcome(j)+j.detail)+'</p></div><div class="acts">'+
      (j.replay ? '<button type="button" class="soft" data-trial-video="'+esc(j.id)+'">双摄回放</button>' : '')+
      button('log','日志',j.id)+'</div></div>').join('') + reference || empty('还没有从控制台发起的试跑。'));
  }

  async function status() {
    if (!here() || !token || reading) return;
    reading = true;
    try {
      const next = await json('/api/trial-status',{method:'POST',headers:postHeaders(),
        body:JSON.stringify({heartbeat: waitingId || (trial.owned ? trial.id : '')}),signal:AbortSignal.timeout(1800)});
      if (waitingId && next.id===waitingId) waitingId='';
      if (trial.status==='running' && next.status!=='running') void poll();
      trial=next; reachable=true;
    } catch { reachable=false; }
    finally { reading=false; draw(); }
  }
  async function act(name) {
    if (el(name).disabled || el(name).hidden) return;
    const jobId=trial.id;
    if (name==='release' && !confirm('已用手或稳固支撑托住从臂，并放稳夹持物？确认后六个关节将卸力。')) return;
    pending=name; draw();
    try {
      const result=await json('/api/action',{method:'POST',headers:postHeaders(),
        body:JSON.stringify({action:'trial_'+name,job_id:jobId,zone:el('zone').value,model:el('model').value,
          onsite:name==='start'&&el('onsite').checked,supported:name==='release'}),
        signal:AbortSignal.timeout(3500)});
      if (name==='check') waitingId=result.job_id;
      await status();
      void poll();
    } catch(e) { toast(e.message,true); await status(); }
    finally { pending=''; draw(); }
  }
  actions.forEach(name=>el(name).addEventListener('click',()=>act(name)));
  el('onsite').addEventListener('change',draw);
  el('model').addEventListener('change',draw);
  el('history').addEventListener('click',e=>{
    const b=e.target.closest('[data-trial-video]');
    if (!b) return;
    const video=el('video'), id=b.dataset.trialVideo;
    if (id!==videoId) { video.src='/api/trial-replay?id='+encodeURIComponent(id); videoId=id; }
    video.hidden=false; video.play().catch(()=>{});
  });
  document.addEventListener('visibilitychange',status);
  window.addEventListener('hashchange',status);
  draw(); status(); setInterval(status,650);
})();
