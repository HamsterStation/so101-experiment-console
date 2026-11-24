/* Planning is local. A proposed episode index becomes real only after the operator matches it to a saved run. */
(() => {
  'use strict';
  const el=id=>document.getElementById(id);
  const zones=window.so101ZonePlan;
  const rows=[];
  // Rotate the first cell of each round; spread held-out placements across the session.
  const reserve=[[1,6],[3,7],[4],[2],[2,8],[1,5]];
  const seen=zones.map(()=>0);
  for(let round=0;round<9;round++)for(let n=0;n<6;n++){
    const i=(n+round)%6,z=zones[i];if(seen[i]>=z.n)continue;
    const within=++seen[i];
    rows.push({id:rows.length,zone:z.id,within,held:reserve[i].includes(within)});
  }
  const valIds=rows.filter(r=>r.held).map(r=>r.id);
  const str=v=>String(v??'未提供');
  const panels=['library','plan','public','flow'];
  const E=(name,txt)=>{const e=document.createElement(name);if(txt!=null)e.textContent=str(txt);return e};
  function showPanel(p) {
    p=panels.includes(p)?p:'library';
    document.querySelectorAll('[data-dstab]').forEach(a=>{if(a.dataset.dstab===p)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current')});
    document.querySelectorAll('[data-dspanel]').forEach(s=>s.hidden=s.dataset.dspanel!==p);
  }
  el('episode-plan').replaceChildren(...rows.map(r=>{const tr=E('tr');[r.id,r.zone,r.within,r.held?'留出验证':'训练'].forEach(x=>tr.append(E('td',x)));return tr}));
  window.initCollectionSpace(el('collection-space'));
  el('plan-export').addEventListener('click',()=>{
    const fields=['proposed_episode_index','start_zone','zone_trial','planned_split','actual_dataset','actual_episode_index','qc_pass','notes'];
    const text='\uFEFF'+[fields.join(','),...rows.map(r=>[r.id,r.zone,r.within,r.held?'validation':'train','','','',''].join(','))].join('\r\n')+'\r\n';
    const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'})),a=E('a');a.href=url;a.download='so101-cube-50-collection-plan.csv';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);
  });
  const result=el('hub-result'),lookup=el('hub-inspect');
  let version=0;
  async function inspect(repo){
    const my=++version;
    el('hub-id').value=repo;
    result.className='hub-check';result.textContent='正在读取公开元数据…';
    try{
      const response=await fetch('/api/public-dataset?repo='+encodeURIComponent(repo),{cache:'no-store',signal:AbortSignal.timeout(19000)});
      const d=await response.json();if(my!==version)return;
      if(!response.ok)throw Error(d.error||'未能取得元数据');
      const schemes=d.cameras.map(k=>k.replace(/^observation\.images\./,''));
      result.replaceChildren(E('strong',d.repo),E('p',[d.version,d.robot,d.episodes+' 条',d.fps+' FPS','许可：'+str(d.license)].join(' · ')),
        E('p','视频/图像字段：'+(schemes.join('、')||'未找到')+'。'+(d.joints_ok?'六维状态、动作与本台的关节名称和顺序相同。':'六维状态或动作的名称、顺序或形状不一致或缺失。')),
        E('p','适配判断：'+(schemes.length!==2?'当前配方需要两个视角；先挑选或补齐视角。':!d.joints_ok?'先核对并转换关节定义。':'字段数量可对照；仍需对照视频确认正面／腕部对应关系、角度单位、动作方向和频率。')+' 本台为 20 FPS；只有完成适配并在本地校验后才能进入训练。'));
      const a=E('a','已检查版本 '+d.revision.slice(0,8)+' ↗');a.href=d.meta_url;a.target='_blank';a.rel='noopener';result.append(a);
    }catch(e){if(my===version)result.textContent=e.name==='TimeoutError'?'查询超时，可稍后重试或打开数据卡。':str(e.message);}
  }
  lookup.addEventListener('submit',e=>{e.preventDefault();inspect(el('hub-id').value.trim())});
  document.querySelectorAll('[data-repo]').forEach(b=>b.addEventListener('click',()=>{inspect(b.dataset.repo);lookup.scrollIntoView({block:'center'})}));
  let list=[];
  function datasetSelect(){
    const d=list.find(x=>x.id===el('flow-data').value);
    el('flow-summary').textContent=d?[(d.synthetic?'合成演练，仅用于验证流程':'实录演示'),d.detail].join(' · '):'请先在本地目录将一份通过检查的双摄录制传到 4080。';
    document.querySelectorAll('[data-flowmodel]').forEach(b=>b.disabled=!d);
  }
  function update(s){
    const inventory=s.datasets||[],real=inventory.filter(d=>d.location==='jetson'&&!d.synthetic);
    el('real-recordings').textContent=s.devices?.jetson?.online?'Jetson 目录已登记实录：'+real.reduce((t,d)=>t+(d.episodes||0),0)+' 条（未按任务或分区质检，不含合成演练和 4080 副本）。':'Jetson 实录数量等待设备同步。';
    list=inventory.filter(d=>d.location==='gpu4080');
    const select=el('flow-data'),old=select.value;
    const key=JSON.stringify(list.map(d=>[d.id,d.name,d.synthetic]));
    if(select.dataset.sig!==key){
      select.dataset.sig=key;select.replaceChildren(...(list.length?list.map(d=>new Option(d.name+(d.synthetic?'（合成）':''),d.id)):[new Option('暂无可用数据集','')]));
      if(list.some(d=>d.id===old))select.value=old;
    }
    datasetSelect();
    const done=(s.jobs||[]).find(j=>j.kind==='train'&&j.synthetic&&j.status==='succeeded'&&j.split&&j.split!=='all'),proof=el('split-example');
    if(done&&proof.dataset.run!==done.id){
      const a=E('a','查看这次合成演练的实际划分和配方 ↗');a.href='/api/manifest?id='+encodeURIComponent(done.id);a.target='_blank';a.rel='noopener';proof.replaceChildren(a);proof.dataset.run=done.id;
    }
  }
  el('flow-data').addEventListener('change',datasetSelect);
  document.querySelectorAll('[data-flowmodel]').forEach(b=>b.addEventListener('click',()=>{
    const d=list.find(x=>x.id===el('flow-data').value);if(!d)return;
    el('train-data').value=d.id;el('recipes').value=b.dataset.flowmodel;el('split-mode').value=d.episodes>=2?'episode80':'all';el('split-ids').value='';
    location.hash='training';el('train-form').dispatchEvent(new Event('change'));
  }));
  el('apply-plan').addEventListener('click',()=>{
    if(confirm('确认已逐条核对：同一份 50 条实录数据集中的实际编号与采集单完全一致？')){el('split-ids').value=valIds.join(',');el('train-form').dispatchEvent(new Event('change'))}
  });
  let identity;
  function splitState(dataset){
    const key=dataset?.id??'';
    if(key!==identity){if(identity!=null)el('split-ids').value='';identity=key}
    const mode=el('split-mode').value,ids=el('split-ids').value.trim(),n=dataset?.episodes||0;
    el('split-manual').hidden=mode!=='manual';el('apply-plan').hidden=!(n===50&&!dataset.synthetic);
    let error='',detail='';
    if(n&&mode!=='all'&&n<2)error='至少需要两条独立演示才能留出验证。单条只适合检查流程。';
    else if(mode==='manual' && n){
      const nums=ids.split(',').map(s=>Number(s.trim()));
      if(!/^\d+(?:\s*,\s*\d+)*$/.test(ids)||new Set(nums).size!==nums.length||nums.some(x=>!Number.isSafeInteger(x)||x>=n)||nums.length>=n)error='请填写独立、无重复且在 0–'+(n-1)+' 内的留出编号，并保留至少一条训练演示。';
      else detail='按你指定的编号留出 '+nums.length+' 条，训练 '+(n-nums.length)+' 条。';
    }else if(n){detail=mode==='all'?'全部 '+n+' 条进入训练；此运行没有独立留出集。':'训练和验证按完整轨迹隔离。随机划分不使用分区标签。';}
    el('split-feedback').textContent=error||detail;
    return {ok:!error,split:mode,validation_episodes:mode==='manual'?ids.split(',').map(s=>Number(s.trim())):[]};
  }
  window.dataWorkbench={showPanel,update,splitState};
})();
