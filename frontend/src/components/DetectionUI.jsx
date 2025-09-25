// src/components/DetectionUI.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { predict, health } from "../lib/api";

const PALETTE = ["#50C2F8","#22D3A6","#A78BFA","#FB7185","#F59E0B",
                 "#60A5FA","#22D3EE","#10B981","#E879F9","#F472B6"];

const colorOf = i => PALETTE[i % PALETTE.length];

/* ---------- RESILIENT ACCESSORS (fix) ---------- */
// accept either {bbox:[x1,y1,x2,y2]} or {boxes:[x1,y1,x2,y2]}
function getBox(inst) {
  if (Array.isArray(inst?.bbox) && inst.bbox.length === 4) return inst.bbox;
  if (Array.isArray(inst?.boxes) && inst.boxes.length === 4) return inst.boxes;
  return [0,0,0,0];
}
// accept either {mask: HxW} or {masks: HxW}
function getMask(inst) {
  const m = inst?.mask ?? inst?.masks;
  if (!m) return null;
  // ensure it's a 2D array
  if (Array.isArray(m) && Array.isArray(m[0])) return m;
  return null;
}
// unify truthiness for mask element
const onPix = v => v === 1 || v === 255 || v === true;

/* ---------- GEOMETRY + TYPING ---------- */
function geomFromInstance(inst) {
  const [x1,y1,x2,y2] = getBox(inst);
  const w = Math.max(0,(x2-x1)|0), h = Math.max(0,(y2-y1)|0);
  const M = getMask(inst);

  if (M) {
    const H = M.length, W = M[0].length;
    let area = 0, perim = 0, minx=W, miny=H, maxx=0, maxy=0;
    for (let y=0; y<H; y++) {
      const row = M[y];
      for (let x=0; x<W; x++) {
        if (onPix(row[x])) {
          area++;
          if (x<minx) minx=x; if (x>maxx) maxx=x;
          if (y<miny) miny=y; if (y>maxy) maxy=y;
          const up    = (y===0)   || !onPix(M[y-1][x]);
          const down  = (y===H-1) || !onPix(M[y+1][x]);
          const left  = (x===0)   || !onPix(M[y][x-1]);
          const right = (x===W-1) || !onPix(M[y][x+1]);
          perim += (up?1:0) + (down?1:0) + (left?1:0) + (right?1:0);
        }
      }
    }
    if (area === 0) return { area_px:0, width:0, height:0, bbox:[0,0,0,0], aspect:0, roundness:0 };

    const bw = Math.max(1,(maxx-minx+1)|0), bh = Math.max(1,(maxy-miny+1)|0);
    const aspect = Math.max(bw,bh) / Math.max(1, Math.min(bw,bh));
    const roundness = perim>0 ? (4*Math.PI*area)/(perim*perim) : 0;
    return { area_px:area, width:bw, height:bh, bbox:[minx,miny,maxx+1,maxy+1], aspect, roundness };
  }

  // fallback to detector's box if mask missing
  return { area_px:w*h, width:w, height:h, bbox:[x1,y1,x2,y2], aspect:(w&&h?Math.max(w,h)/Math.min(w,h):0), roundness:0 };
}

function classify(shape) {
  const { area_px, aspect, roundness, width, height } = shape;
  if (area_px <= 0) return "Other";
  if (aspect >= 4 && roundness < 0.35) return "Crack";
  if (aspect >= 2.5 && roundness < 0.5 && Math.min(width,height) <= 16) return "Scratch";
  if (roundness >= 0.6 && area_px <= 400) return "Spot";
  if (roundness < 0.5 && area_px > 400) return "Stain";
  if (aspect >= 1.4 && area_px > 700) return "Peeling Paint";
  return "Other";
}
function cure(type){
  switch(type){
    case "Crack": return "Stabilize; micro-fill; inpaint with reversible media.";
    case "Scratch": return "Local fill if visible; retouch under magnification.";
    case "Peeling Paint": return "Consolidate loose paint; lay & set under heat/spatula.";
    case "Stain": return "Spot-clean with tested solvent/gel; proceed gradually.";
    case "Spot": return "Soft-brush or air bulb; lift with swab if bonded.";
    default: return "Document, evaluate under raking/UV; consult a conservator.";
  }
}

export default function DetectionUI(){
  const [file,setFile] = useState(null);
  const [imgURL,setImgURL] = useState(null);
  const [out,setOut] = useState(null);
  const [scoreTh,setScoreTh] = useState(0.5);
  const [loading,setLoading] = useState(false);
  const [status,setStatus] = useState("");
  const [err,setErr] = useState(null);

  const [drawerOpen,setDrawerOpen] = useState(true);
  const [hoverIdx,setHoverIdx] = useState(null);
  const [flashIdx,setFlashIdx] = useState(null);

  // zoom/pan
  const [zoom,setZoom] = useState(1);
  const [pan,setPan] = useState({x:0,y:0});
  const dragRef = useRef(false), lastRef = useRef({x:0,y:0});
  const canvasRef = useRef(null);

  useEffect(()=>{
    if(!file){ setImgURL(null); return; }
    const url = URL.createObjectURL(file);
    setImgURL(url);
    return ()=> URL.revokeObjectURL(url);
  },[file]);

  useEffect(()=>{
    (async()=>{ try{ const h=await health(); setStatus(`API: ${JSON.stringify(h)}`); } catch(e){ setStatus(`API health failed: ${e.message}`);} })();
  },[]);

  async function onRun(){
    if(!file){ setStatus("No file selected"); return; }
    setLoading(true); setErr(null); setStatus("Detecting…"); setOut(null);
    try{ const res = await predict(file); setOut(res); setStatus(`Done. Instances: ${res?.instances?.length ?? 0}`); setZoom(1); setPan({x:0,y:0}); }
    catch(e){ setErr(e.message||"Request failed"); setStatus(`Error: ${e.message||e}`); }
    finally{ setLoading(false); }
  }
  async function useSample(){
    try{ const r=await fetch("/sample.jpg"); const b=await r.blob(); setFile(new File([b],"sample.jpg",{type:b.type||"image/jpeg"})); setStatus("Loaded sample.jpg"); }
    catch{ setStatus("Put sample.jpg in frontend/public/"); }
  }

  // draw canvas with masks and boxes (using fixed accessors)
  useEffect(()=>{
    const c = canvasRef.current;
    if(!c || !out?.preview_base64) return;

    const ctx = c.getContext("2d");
    const base = new Image();
    base.onload = ()=>{
      c.width = base.width; c.height = base.height;
      ctx.setTransform(1,0,0,1,0,0);
      ctx.clearRect(0,0,c.width,c.height);
      ctx.translate(pan.x, pan.y);
      ctx.scale(zoom, zoom);
      ctx.drawImage(base,0,0);

      const vis = (out.instances||[]).filter(i => (i.score??1) >= scoreTh);
      vis.forEach(d=>{
        const idx = out.instances.indexOf(d);
        const color = colorOf(idx);
        const glow = (hoverIdx===idx) || (flashIdx===idx);
        const M = getMask(d);

        if (M){ // paint soft mask
          const H=M.length, W=M[0].length;
          const id = ctx.createImageData(W,H);
          const R=parseInt(color.slice(1,3),16),
                G=parseInt(color.slice(3,5),16),
                B=parseInt(color.slice(5,7),16);
          let k=0;
          for(let y=0;y<H;y++) for(let x=0;x<W;x++){
            const on = onPix(M[y][x]);
            id.data[k++]=R; id.data[k++]=G; id.data[k++]=B; id.data[k++]= on ? (glow?96:64) : 0;
          }
          ctx.putImageData(id,0,0);
        }

        const [x1,y1,x2,y2] = getBox(d);
        if (x2>x1 && y2>y1){
          if (glow){ ctx.save(); ctx.shadowColor=color; ctx.shadowBlur=22; ctx.strokeStyle=color; ctx.lineWidth=4; ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.restore(); }
          ctx.strokeStyle=color; ctx.lineWidth=2; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
          ctx.fillStyle=glow?color:color+"66";
          ctx.fillRect(x1,y1-18,150,18);
          ctx.fillStyle="#081018"; ctx.font="12px system-ui,-apple-system,Segoe UI,Roboto,Arial";
          ctx.fillText(`defect ${idx+1} (${(d.score??0).toFixed(2)})`,x1+6,y1-5);
        }
      });
      if (flashIdx!==null) setTimeout(()=>setFlashIdx(null),180);
    };
    base.src = `data:image/jpeg;base64,${out.preview_base64}`;
  },[out,scoreTh,hoverIdx,flashIdx,zoom,pan]);

  // enrich for panel
  const defects = useMemo(()=>{
    if (!out?.instances) return [];
    return out.instances
      .map((inst,i)=>{
        const shape = geomFromInstance(inst);
        const type  = classify(shape);
        return { __idx:i, score:inst.score ?? inst.scores ?? 0, ...shape, box:getBox(inst), type, cure:cure(type) };
      })
      .filter(d => (d.score ?? 1) >= scoreTh);
  },[out,scoreTh]);

  // interactions
  function onWheel(e){ if(!out) return; e.preventDefault(); setZoom(z=>Math.min(5,Math.max(0.2,z*(e.deltaY<0?1.1:0.9)))); }
  function onDown(e){ dragRef.current=true; lastRef.current={x:e.clientX,y:e.clientY}; }
  function onMove(e){ if(!dragRef.current) return; const dx=e.clientX-lastRef.current.x, dy=e.clientY-lastRef.current.y; lastRef.current={x:e.clientX,y:e.clientY}; setPan(p=>({x:p.x+dx,y:p.y+dy})); }
  function onUp(){ dragRef.current=false; }

  // export
  function exportCSV(){
    if(!defects.length) return;
    const head=["id","score","type","area_px","width","height","x1","y1","x2","y2","cure"];
    const rows=defects.map(d=>[d.__idx+1,d.score.toFixed(3),d.type,d.area_px,d.width,d.height,d.box[0],d.box[1],d.box[2],d.box[3],`"${d.cure.replace(/"/g,'""')}"`].join(","));
    const csv=[head.join(","),...rows].join("\n");
    const blob=new Blob([csv],{type:"text/csv;charset=utf-8"});
    const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="defects.csv"; a.click(); URL.revokeObjectURL(a.href);
  }
  function exportPNG(){ const c=canvasRef.current; if(!c) return; const a=document.createElement("a"); a.href=c.toDataURL("image/png"); a.download="defects_annotated.png"; a.click(); }

  return (
    <>
      <div className="toolbar glass">
        <div className="tinline">
          <label className="fileBtn">
            <input type="file" accept="image/*" onChange={e=>e.target.files?.[0] && setFile(e.target.files[0])}/>
            Choose image
          </label>
          <button className="btn" onClick={useSample}>Use Sample</button>
        </div>
        <div className="tinline">
          <button className="btn accent" onClick={onRun} disabled={!file||loading}>{loading?"Detecting…":"Run Detection"}</button>
          <button className="btn ghost" onClick={()=>{ setOut(null); setImgURL(null); setFile(null); setErr(null); setStatus(""); setZoom(1); setPan({x:0,y:0}); }}>Clear</button>
        </div>
        <div className="tinline">
          <div className="th"><span>Score ≥</span><span className="thbubble">{scoreTh.toFixed(2)}</span>
            <input type="range" min="0" max="0.95" step="0.05" value={scoreTh} onChange={e=>setScoreTh(parseFloat(e.target.value))}/>
          </div>
          <button className="btn" onClick={()=>setDrawerOpen(s=>!s)}>{drawerOpen?"Hide Defects":"Show Defects"}</button>
          <button className="btn" onClick={exportCSV} disabled={!defects.length}>Export CSV</button>
          <button className="btn" onClick={exportPNG} disabled={!out?.preview_base64}>Export PNG</button>
        </div>
      </div>

      {status && <div className="status">{status}</div>}
      {err && <div className="err">{err}</div>}

      <div className="stage">
        <div className="panel glass">
          <div className="phd">Input</div>
          <div className="pbd">{imgURL ? <img className="fit shadow" src={imgURL} alt="input"/> : <div className="ph">Choose an image or click “Use Sample”</div>}</div>
        </div>
        <div className="panel glass">
          <div className="phd">Output <span className="hint">• wheel: zoom • drag: pan</span></div>
          <div className="pbd">
            {out?.preview_base64 ? (
              <canvas ref={canvasRef} className="fit shadow"
                onWheel={onWheel} onMouseDown={onDown} onMouseMove={onMove}
                onMouseUp={onUp} onMouseLeave={onUp}/>
            ) : <div className="ph">Run detection</div>}
          </div>
        </div>

        <aside className={drawerOpen ? "drawer open glass" : "drawer glass"}>
          <div className="phd">Defects</div>
          <div className="list">
            {!defects.length ? <div className="ph">No items ≥ threshold</div> :
              defects.map(d=>{
                const color=colorOf(d.__idx);
                const [x1,y1,x2,y2]=d.box;
                return (
                  <div key={d.__idx}
                       className={"cardrow"+(hoverIdx===d.__idx?" active":"")}
                       onMouseEnter={()=>setHoverIdx(d.__idx)}
                       onMouseLeave={()=>setHoverIdx(null)}
                       onClick={()=>setFlashIdx(d.__idx)}>
                    <div className="sw" style={{background:color}} />
                    <div className="meta">
                      <div className="lb">defect #{d.__idx+1} • {d.type}</div>
                      <div className="sb">score {d.score.toFixed(2)} • {d.width}×{d.height}px • area {d.area_px}px²</div>
                      <div className="sb" style={{opacity:.9}}>bbox [{x1},{y1}]–[{x2},{y2}]</div>
                      <div className="sb" style={{opacity:.9}}>cure: {d.cure}</div>
                    </div>
                  </div>
                );
              })}
          </div>
        </aside>
      </div>
    </>
  );
}
