/* Collection-space example. All counts are design targets, independent of recorded data. */
(() => {
  'use strict';
  const zones = [
    {id:'A1', n:9, val:2}, {id:'A2', n:8, val:2}, {id:'A3', n:8, val:1},
    {id:'B1', n:8, val:1}, {id:'B2', n:9, val:2}, {id:'B3', n:8, val:2},
  ];
  window.so101ZonePlan = Object.freeze(zones.map(z=>Object.freeze({...z})));
  const total = zones.reduce((sum, z) => sum + z.n, 0);
  const valid = zones.reduce((sum, z) => sum + z.val, 0);
  const side = ['左侧', '中部', '右侧'];
  let seq = 0;

  window.initCollectionSpace = function initCollectionSpace(host) {
    if (!host || host.querySelector('[data-space-diagram]')) return;
    const uid = 'space-example-' + (++seq);
    const cells = zones.map((z,i) => {
      const x=48+(i%3)*72, y=36+Math.floor(i/3)*58;
      return '<g class="cs-zone" role="radio" tabindex="'+(i===0?'0':'-1')+'" aria-checked="'+(i===0)+'" aria-label="'+z.id+'，计划 '+z.n+' 条，其中训练 '+(z.n-z.val)+' 条、验证 '+z.val+' 条" data-zone="'+i+'" transform="translate('+x+' '+y+')"><rect class="cs-field" width="66" height="51" rx="5"/><text class="cs-loc" x="8" y="20">'+z.id+'</text><text class="cs-quota" x="8" y="39">'+z.n+' 条</text><path class="cs-cube" d="m50 9 6-3 6 3v7l-6 3-6-3z M50 9l6 3 6-3m-6 3v7"/></g>';
    }).join('');
    host.innerHTML = `<section class="cs-widget" aria-labelledby="${uid}-heading" data-space-diagram>
      <header class="cs-head"><div><h3 id="${uid}-heading">桌面采集分区</h3><p>固定目标托盘，只改变方块的起点。</p></div><span class="cs-example">规划示例</span></header>
      <div class="cs-layout">
        <figure class="cs-figure">
          <svg class="cs-map" viewBox="0 0 410 186" role="group" aria-labelledby="${uid}-title" aria-describedby="${uid}-desc">
            <title id="${uid}-title">SO-101 桌面俯视示意图，可选择六个方块起点分区</title>
            <desc id="${uid}-desc">机器人底座位于图的下方。A 排在远侧，B 排在近侧。右方为固定目标托盘。按方向键切换起点分区。</desc>
            <rect class="cs-table" x="1" y="1" width="408" height="184" rx="9"/>
            <text class="cs-map-label" x="48" y="24">方块起点</text>
            <text class="cs-map-side" x="14" y="66">远</text><text class="cs-map-side" x="14" y="125">近</text>
            <g role="radiogroup" aria-label="查看各区采集配额">${cells}</g>
            <rect class="cs-tray" x="294" y="57" width="85" height="65" rx="8"/>
            <rect class="cs-tray-in" x="299" y="62" width="75" height="55" rx="5"/>
            <text class="cs-tray-name" x="336.5" y="95">目标托盘</text>
            <path class="cs-base-mark" d="M114 172h12m8-4v8m-4-4h8"/>
            <text class="cs-map-label" x="144" y="176">底座一侧</text>
          </svg>
          <figcaption>俯视示意，无尺寸比例。实际分区以夹爪可达、双摄可见的桌面为准。</figcaption>
        </figure>
        <div class="cs-info" aria-live="polite" aria-atomic="true">
          <div class="cs-selected"></div>
          <p class="cs-action"></p>
          <dl><div><dt>训练</dt><dd data-space-train></dd></div><div><dt>验证</dt><dd data-space-val></dd></div></dl>
        </div>
      </div>
      <div class="cs-totals"><b>全桌计划 ${total} 条</b><span>训练 ${total-valid} 条 · 验证 ${valid} 条</span></div>
      <p class="cs-note">一条演示包含从抓起方块到放入托盘的完整过程。每区另摆起点录验证演示；同一条演示的所有帧只归一组。图中数字为计划配额，实际数量见数据集目录。</p>
    </section>`;
    const buttons = [...host.querySelectorAll('[data-zone]')];
    function choose(i, focus) {
      const z=zones[i];
      buttons.forEach((b,j) => {
        b.setAttribute('aria-checked',String(j===i));
        b.setAttribute('tabindex',j===i?'0':'-1');
      });
      host.querySelector('.cs-selected').textContent=z.id+' · '+(i<3?'远排':'近排')+' · '+side[i%3];
      host.querySelector('.cs-action').textContent='在 '+z.id+' 区内每次稍微改变方块位置，完成 '+z.n+' 次成功抓取放置。';
      host.querySelector('[data-space-train]').textContent=(z.n-z.val)+' 条';
      host.querySelector('[data-space-val]').textContent=z.val+' 条';
      if(focus)buttons[i].focus();
    }
    buttons.forEach((button,i)=>{
      button.addEventListener('click',()=>choose(i,true));
      button.addEventListener('keydown',ev=>{
        let j=i;
        switch(ev.key){
          case 'ArrowLeft': j=(i+5)%6; break;
          case 'ArrowRight': j=(i+1)%6; break;
          case 'ArrowUp': case 'ArrowDown': j=(i+3)%6; break;
          case 'Home': j=0; break;
          case 'End': j=5; break;
          case ' ': case 'Enter': break;
          default: return;
        }
        ev.preventDefault(); choose(j,true);
      });
    });
    choose(0,false);
  };
})();
