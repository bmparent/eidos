import { guarded } from '../../_shared/platform/core';
import { handleHost, hostStatus } from '../../_shared/platform/exchangeHost';
export const onRequestGet = guarded(hostStatus);
export const onRequestPost = guarded(handleHost);
