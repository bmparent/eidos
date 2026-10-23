// Loaded only by the explicit Enable Qwen action. Model inference stays on device.
import {MODEL_SYSTEM} from '/browser-interpreter.mjs';
const MODEL='onnx-community/Qwen3-0.6B-ONNX';
const REVISION='58ca03a5e7b354b480b1a82b34bde8dc7b854cc0';
const RUNTIME='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1/dist/transformers.web.min.js';
let tokenizer,model,busy=false;
async function load() {
  const {AutoTokenizer,AutoModelForCausalLM,env}=await import(RUNTIME);
  env.allowLocalModels=false;
  // Single-thread WASM avoids cross-origin-isolation requirements; inference is WebGPU.
  env.backends.onnx.wasm.numThreads=1;
  const progress_callback=report=>{
    if(report.status==='progress')self.postMessage({type:'progress',progress:report.progress,reason:`Downloading ${report.file||'Qwen weights'} · ${Math.round(report.progress||0)}%`});
    else if(report.status==='initiate')self.postMessage({type:'progress',progress:null,reason:'Downloading Qwen files for this device…'});
  };
  [tokenizer,model]=await Promise.all([
    AutoTokenizer.from_pretrained(MODEL,{revision:REVISION,progress_callback}),
    AutoModelForCausalLM.from_pretrained(MODEL,{revision:REVISION,dtype:'q4f16',device:'webgpu',progress_callback}),
  ]);
  self.postMessage({type:'progress',progress:100,reason:'Compiling Qwen shaders…'});
  await model.generate({...tokenizer('a'),max_new_tokens:1});
  return {model:MODEL,runtime:RUNTIME};
}
async function rank({prompt}) {
  if(!tokenizer||!model)throw new Error('Qwen is not loaded.');
  if(typeof prompt!=='string'||prompt.length>6000)throw new Error('Model context was not bounded.');
  const inputs=tokenizer.apply_chat_template([
    {role:'system',content:MODEL_SYSTEM},
    {role:'user',content:prompt},
  ],{add_generation_prompt:true,return_dict:true,enable_thinking:false});
  const started=performance.now();
  const output=await model.generate({...inputs,max_new_tokens:40,do_sample:false,repetition_penalty:1.1,return_dict_in_generate:true});
  const promptLength=inputs.input_ids.dims.at(-1);
  const generated=output.sequences.slice(null,[promptLength,null]);
  const raw=tokenizer.batch_decode(generated,{skip_special_tokens:true})[0];
  return {raw,inference_ms:Math.round(performance.now()-started),generated_tokens:generated.dims.at(-1)};
}
self.onmessage=async({data})=>{
  const {id,type}=data;
  if(busy){self.postMessage({id,error:'One local model job is already running.'});return;}
  busy=true;
  try{const result=type==='load'?await load():type==='rank'?await rank(data.data):null;if(result===null)throw new Error('Unknown model operation.');self.postMessage({id,result});}
  catch{self.postMessage({id,error:type==='load'?'Qwen could not load. Check device WebGPU support, available memory, and access to Hugging Face/jsDelivr model downloads.':'Qwen inference failed. The original measurement remains available.'});}
  finally{busy=false;}
};
