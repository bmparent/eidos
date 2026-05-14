export function trackOrganisms({alive,age,energy,stress,width,height,previous=[]}){
  const visited=new Uint8Array(alive.length); const organisms=[];
  const idx=(x,y)=>((y+height)%height)*width+((x+width)%width);
  const usedIds = new Set();
  let nextId = previous.reduce((max,p)=>Math.max(max,p.id||0),0)+1;
  for(let y=0;y<height;y++) for(let x=0;x<width;x++){
    const i=idx(x,y); if(!alive[i]||visited[i]) continue;
    const stack=[[x,y]]; let cells=[],sx=0,sy=0,a=0,e=0,s=0,minX=1e9,maxX=-1,minY=1e9,maxY=-1;
    while(stack.length){ const [cx,cy]=stack.pop(); const ci=idx(cx,cy); if(visited[ci]||!alive[ci]) continue; visited[ci]=1; cells.push(ci);
      sx+=cx; sy+=cy; a+=age[ci]; e+=energy[ci]; s+=stress[ci]; minX=Math.min(minX,cx); maxX=Math.max(maxX,cx); minY=Math.min(minY,cy); maxY=Math.max(maxY,cy);
      [[1,0],[-1,0],[0,1],[0,-1]].forEach(([dx,dy])=>{ const ni=idx(cx+dx,cy+dy); if(!visited[ni]&&alive[ni]) stack.push([cx+dx,cy+dy]); });
    }
    const mass=cells.length, centroid={x:sx/mass,y:sy/mass};
    let best=null,bestScore=1e9; for(const p of previous){ const d=Math.hypot(p.centroid.x-centroid.x,p.centroid.y-centroid.y)+Math.abs((p.mass||0)-mass)*0.3; if(d<bestScore){best=p;bestScore=d;} }
    let id = best&&bestScore<8 ? best.id : null;
    if (id == null || usedIds.has(id)){
      while (usedIds.has(nextId)) nextId++;
      id = nextId++;
    }
    usedIds.add(id);
    organisms.push({id,centroid,mass,boundingBox:{minX,maxX,minY,maxY},meanAge:a/mass,meanEnergy:e/mass,meanStress:s/mass,lifespanEstimate:Math.max(1,a/mass*1.8),noveltyScore:Math.min(1,(maxX-minX+maxY-minY)/(width+height)),threatScore:Math.min(1,s/mass + mass/180)});
  }
  return organisms;
}
