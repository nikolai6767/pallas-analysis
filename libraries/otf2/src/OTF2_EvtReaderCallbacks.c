#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "otf2/OTF2_EvtReaderCallbacks.h"
#include "otf2/otf2.h"

OTF2_EvtReaderCallbacks* OTF2_EvtReaderCallbacks_New(void) {

  OTF2_EvtReaderCallbacks* ret = malloc(sizeof(OTF2_EvtReaderCallbacks));
  OTF2_EvtReaderCallbacks_Clear(ret);
  return ret;
}

void OTF2_EvtReaderCallbacks_Delete(OTF2_EvtReaderCallbacks* evtReaderCallbacks) {
  free(evtReaderCallbacks);
}

void OTF2_EvtReaderCallbacks_Clear(OTF2_EvtReaderCallbacks* evtReaderCallbacks) {
  memset(evtReaderCallbacks, 0, sizeof(OTF2_EvtReaderCallbacks));
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetUnknownCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_Unknown unknownCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_Unknown_callback = unknownCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetBufferFlushCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                              OTF2_EvtReaderCallback_BufferFlush bufferFlushCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_BufferFlush_callback = bufferFlushCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMeasurementOnOffCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_MeasurementOnOff measurementOnOffCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MeasurementOnOff_callback = measurementOnOffCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetEnterCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                        OTF2_EvtReaderCallback_Enter enterCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_Enter_callback = enterCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetLeaveCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                        OTF2_EvtReaderCallback_Leave leaveCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_Leave_callback = leaveCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiSendCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_MpiSend mpiSendCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiSend_callback = mpiSendCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiIsendCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                           OTF2_EvtReaderCallback_MpiIsend mpiIsendCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiIsend_callback = mpiIsendCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiIsendCompleteCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_MpiIsendComplete mpiIsendCompleteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiIsendComplete_callback = mpiIsendCompleteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiIrecvRequestCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_MpiIrecvRequest mpiIrecvRequestCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiIrecvRequest_callback = mpiIrecvRequestCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiRecvCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_MpiRecv mpiRecvCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiRecv_callback = mpiRecvCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiIrecvCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                           OTF2_EvtReaderCallback_MpiIrecv mpiIrecvCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiIrecv_callback = mpiIrecvCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiRequestTestCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_MpiRequestTest mpiRequestTestCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiRequestTest_callback = mpiRequestTestCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiRequestCancelledCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_MpiRequestCancelled mpiRequestCancelledCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiRequestCancelled_callback = mpiRequestCancelledCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiCollectiveBeginCallback(
								     OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								     OTF2_EvtReaderCallback_MpiCollectiveBegin mpiCollectiveBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiCollectiveBegin_callback = mpiCollectiveBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMpiCollectiveEndCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_MpiCollectiveEnd mpiCollectiveEndCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_MpiCollectiveEnd_callback = mpiCollectiveEndCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpForkCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_OmpFork ompForkCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpFork_callback = ompForkCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpJoinCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_OmpJoin ompJoinCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpJoin_callback = ompJoinCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpAcquireLockCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_OmpAcquireLock ompAcquireLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpAcquireLock_callback = ompAcquireLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpReleaseLockCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_OmpReleaseLock ompReleaseLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpReleaseLock_callback = ompReleaseLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpTaskCreateCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_OmpTaskCreate ompTaskCreateCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpTaskCreate_callback = ompTaskCreateCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpTaskSwitchCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_OmpTaskSwitch ompTaskSwitchCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpTaskSwitch_callback = ompTaskSwitchCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetOmpTaskCompleteCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_OmpTaskComplete ompTaskCompleteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_OmpTaskComplete_callback = ompTaskCompleteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetMetricCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                         OTF2_EvtReaderCallback_Metric metricCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_Metric_callback = metricCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetParameterStringCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_ParameterString parameterStringCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ParameterString_callback = parameterStringCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetParameterIntCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_ParameterInt parameterIntCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ParameterInt_callback = parameterIntCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetParameterUnsignedIntCallback(
								       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								       OTF2_EvtReaderCallback_ParameterUnsignedInt parameterUnsignedIntCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ParameterUnsignedInt_callback = parameterUnsignedIntCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaWinCreateCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_RmaWinCreate rmaWinCreateCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaWinCreate_callback = rmaWinCreateCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaWinDestroyCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_RmaWinDestroy rmaWinDestroyCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaWinDestroy_callback = rmaWinDestroyCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaCollectiveBeginCallback(
								     OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								     OTF2_EvtReaderCallback_RmaCollectiveBegin rmaCollectiveBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaCollectiveBegin_callback = rmaCollectiveBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaCollectiveEndCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_RmaCollectiveEnd rmaCollectiveEndCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaCollectiveEnd_callback = rmaCollectiveEndCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaGroupSyncCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_RmaGroupSync rmaGroupSyncCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaGroupSync_callback = rmaGroupSyncCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaRequestLockCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_RmaRequestLock rmaRequestLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaRequestLock_callback = rmaRequestLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaAcquireLockCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_RmaAcquireLock rmaAcquireLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaAcquireLock_callback = rmaAcquireLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaTryLockCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_RmaTryLock rmaTryLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaTryLock_callback = rmaTryLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaReleaseLockCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_RmaReleaseLock rmaReleaseLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaReleaseLock_callback = rmaReleaseLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaSyncCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                          OTF2_EvtReaderCallback_RmaSync rmaSyncCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaSync_callback = rmaSyncCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaWaitChangeCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_RmaWaitChange rmaWaitChangeCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaWaitChange_callback = rmaWaitChangeCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaPutCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                         OTF2_EvtReaderCallback_RmaPut rmaPutCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaPut_callback = rmaPutCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaGetCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                         OTF2_EvtReaderCallback_RmaGet rmaGetCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaGet_callback = rmaGetCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaAtomicCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                            OTF2_EvtReaderCallback_RmaAtomic rmaAtomicCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaAtomic_callback = rmaAtomicCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaOpCompleteBlockingCallback(
									OTF2_EvtReaderCallbacks* evtReaderCallbacks,
									OTF2_EvtReaderCallback_RmaOpCompleteBlocking rmaOpCompleteBlockingCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaOpCompleteBlocking_callback = rmaOpCompleteBlockingCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaOpCompleteNonBlockingCallback(
									   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
									   OTF2_EvtReaderCallback_RmaOpCompleteNonBlocking rmaOpCompleteNonBlockingCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaOpCompleteNonBlocking_callback = rmaOpCompleteNonBlockingCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaOpTestCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                            OTF2_EvtReaderCallback_RmaOpTest rmaOpTestCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaOpTest_callback = rmaOpTestCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetRmaOpCompleteRemoteCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_RmaOpCompleteRemote rmaOpCompleteRemoteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_RmaOpCompleteRemote_callback = rmaOpCompleteRemoteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadForkCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_ThreadFork threadForkCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadFork_callback = threadForkCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadJoinCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_ThreadJoin threadJoinCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadJoin_callback = threadJoinCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadTeamBeginCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_ThreadTeamBegin threadTeamBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadTeamBegin_callback = threadTeamBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadTeamEndCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_ThreadTeamEnd threadTeamEndCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadTeamEnd_callback = threadTeamEndCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadAcquireLockCallback(
								    OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								    OTF2_EvtReaderCallback_ThreadAcquireLock threadAcquireLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadAcquireLock_callback = threadAcquireLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadReleaseLockCallback(
								    OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								    OTF2_EvtReaderCallback_ThreadReleaseLock threadReleaseLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadReleaseLock_callback = threadReleaseLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadTaskCreateCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_ThreadTaskCreate threadTaskCreateCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadTaskCreate_callback = threadTaskCreateCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadTaskSwitchCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_ThreadTaskSwitch threadTaskSwitchCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadTaskSwitch_callback = threadTaskSwitchCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadTaskCompleteCallback(
								     OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								     OTF2_EvtReaderCallback_ThreadTaskComplete threadTaskCompleteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadTaskComplete_callback = threadTaskCompleteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadCreateCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_ThreadCreate threadCreateCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadCreate_callback = threadCreateCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadBeginCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                              OTF2_EvtReaderCallback_ThreadBegin threadBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadBegin_callback = threadBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadWaitCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_ThreadWait threadWaitCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadWait_callback = threadWaitCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetThreadEndCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                            OTF2_EvtReaderCallback_ThreadEnd threadEndCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ThreadEnd_callback = threadEndCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetCallingContextEnterCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_CallingContextEnter callingContextEnterCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_CallingContextEnter_callback = callingContextEnterCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetCallingContextLeaveCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_CallingContextLeave callingContextLeaveCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_CallingContextLeave_callback = callingContextLeaveCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetCallingContextSampleCallback(
								       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								       OTF2_EvtReaderCallback_CallingContextSample callingContextSampleCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_CallingContextSample_callback = callingContextSampleCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoCreateHandleCallback(
								 OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								 OTF2_EvtReaderCallback_IoCreateHandle ioCreateHandleCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoCreateHandle_callback = ioCreateHandleCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoDestroyHandleCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_IoDestroyHandle ioDestroyHandleCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoDestroyHandle_callback = ioDestroyHandleCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoDuplicateHandleCallback(
								    OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								    OTF2_EvtReaderCallback_IoDuplicateHandle ioDuplicateHandleCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoDuplicateHandle_callback = ioDuplicateHandleCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoSeekCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                         OTF2_EvtReaderCallback_IoSeek ioSeekCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoSeek_callback = ioSeekCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoChangeStatusFlagsCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_IoChangeStatusFlags ioChangeStatusFlagsCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoChangeStatusFlags_callback = ioChangeStatusFlagsCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoDeleteFileCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_IoDeleteFile ioDeleteFileCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoDeleteFile_callback = ioDeleteFileCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoOperationBeginCallback(
								   OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								   OTF2_EvtReaderCallback_IoOperationBegin ioOperationBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoOperationBegin_callback = ioOperationBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoOperationTestCallback(
								  OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								  OTF2_EvtReaderCallback_IoOperationTest ioOperationTestCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoOperationTest_callback = ioOperationTestCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoOperationIssuedCallback(
								    OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								    OTF2_EvtReaderCallback_IoOperationIssued ioOperationIssuedCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoOperationIssued_callback = ioOperationIssuedCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoOperationCompleteCallback(
								      OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								      OTF2_EvtReaderCallback_IoOperationComplete ioOperationCompleteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoOperationComplete_callback = ioOperationCompleteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoOperationCancelledCallback(
								       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								       OTF2_EvtReaderCallback_IoOperationCancelled ioOperationCancelledCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoOperationCancelled_callback = ioOperationCancelledCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoAcquireLockCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_IoAcquireLock ioAcquireLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoAcquireLock_callback = ioAcquireLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoReleaseLockCallback(
								OTF2_EvtReaderCallbacks* evtReaderCallbacks,
								OTF2_EvtReaderCallback_IoReleaseLock ioReleaseLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoReleaseLock_callback = ioReleaseLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetIoTryLockCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                            OTF2_EvtReaderCallback_IoTryLock ioTryLockCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_IoTryLock_callback = ioTryLockCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetProgramBeginCallback(
							       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
							       OTF2_EvtReaderCallback_ProgramBegin programBeginCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ProgramBegin_callback = programBeginCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetProgramEndCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_ProgramEnd programEndCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_ProgramEnd_callback = programEndCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetNonBlockingCollectiveRequestCallback(
									       OTF2_EvtReaderCallbacks* evtReaderCallbacks,
									       OTF2_EvtReaderCallback_NonBlockingCollectiveRequest nonBlockingCollectiveRequestCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_NonBlockingCollectiveRequest_callback = nonBlockingCollectiveRequestCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetNonBlockingCollectiveCompleteCallback(
										OTF2_EvtReaderCallbacks* evtReaderCallbacks,
										OTF2_EvtReaderCallback_NonBlockingCollectiveComplete nonBlockingCollectiveCompleteCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_NonBlockingCollectiveComplete_callback = nonBlockingCollectiveCompleteCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetCommCreateCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                             OTF2_EvtReaderCallback_CommCreate commCreateCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_CommCreate_callback = commCreateCallback;
  return OTF2_SUCCESS;
}

OTF2_ErrorCode OTF2_EvtReaderCallbacks_SetCommDestroyCallback(OTF2_EvtReaderCallbacks* evtReaderCallbacks,
                                                              OTF2_EvtReaderCallback_CommDestroy commDestroyCallback) {
  evtReaderCallbacks->OTF2_EvtReaderCallback_CommDestroy_callback = commDestroyCallback;
  return OTF2_SUCCESS;
}
