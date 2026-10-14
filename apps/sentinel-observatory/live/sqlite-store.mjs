/** Node companion's SQLite implementation of the bounded D1 interface. */
import {DatabaseSync} from 'node:sqlite';
import {readFileSync} from 'node:fs';
export class CompanionDatabase {
  constructor(filename=':memory:') {
    this.sqlite=new DatabaseSync(filename);
    this.sqlite.exec('PRAGMA journal_mode=WAL');
    for(const name of ['0001_observatory_sensor','0002_observatory_model','0003_target_streams'])this.sqlite.exec(readFileSync(new URL('./migrations/'+name+'.sql',import.meta.url),'utf8'));
  }
  prepare(sql) {
    const statement=this.sqlite.prepare(sql);
    return {bind:(...args)=>({first:async()=>statement.get(...args) ?? null,all:async()=>({success:true,results:statement.all(...args)}),
      run:async()=>{const r=statement.run(...args);return {success:true,meta:{changes:r.changes}};},_run:()=>statement.run(...args)})};
  }
  async batch(statements) {
    this.sqlite.exec('BEGIN IMMEDIATE');
    try{const results=statements.map(s=>{const result=s._run();return {success:true,meta:{changes:result.changes}};});this.sqlite.exec('COMMIT');return results;}
    catch(error){this.sqlite.exec('ROLLBACK');throw error;}
  }
  close(){this.sqlite.close();}
}
