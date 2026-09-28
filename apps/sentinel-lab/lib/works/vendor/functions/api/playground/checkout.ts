import {body,guarded,json,origin} from '../../_shared/platform/core';
import {requireMember} from '../../_shared/platform/memberAuth';
import {createPlaygroundCheckout} from '../../_shared/platform/playgroundOrders';
import {ensurePlayground} from '../../_shared/platform/playground';
export const onRequestPost = guarded(async ({request,env}) => {
  origin(request); const member = await requireMember(request,env,false);
  await ensurePlayground(env);
  return json(await createPlaygroundCheckout(env,member.id,await body(request)));
});
