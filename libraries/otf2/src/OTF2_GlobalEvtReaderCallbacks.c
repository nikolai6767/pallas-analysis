#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "otf2/OTF2_GlobalEvtReaderCallbacks.h"
#include "otf2/otf2.h"


/** @brief Allocates a new struct for the event callbacks.
 *
 *  @return A newly allocated struct of type @eref{OTF2_GlobalEvtReaderCallbacks}.
 */
OTF2_GlobalEvtReaderCallbacks* OTF2_GlobalEvtReaderCallbacks_New(void) {
  OTF2_GlobalEvtReaderCallbacks* ret = malloc(sizeof(OTF2_GlobalEvtReaderCallbacks));
  OTF2_GlobalEvtReaderCallbacks_Clear(ret);
  return ret;
}

/** @brief Deallocates a struct for the global event callbacks.
 *
 *  @param globalEvtReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalEvtReaderCallbacks_New}.
 */
void OTF2_GlobalEvtReaderCallbacks_Delete(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks) {
  free(globalEvtReaderCallbacks);
}

/** @brief Clears a struct for the global event callbacks.
 *
 *  @param globalEvtReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalEvtReaderCallbacks_New}.
 */
void OTF2_GlobalEvtReaderCallbacks_Clear(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks) {
  memset(globalEvtReaderCallbacks, 0, sizeof(OTF2_GlobalEvtReaderCallbacks));
}

/** @brief Registers the callback for unknown events.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param unknownCallback          Function which should be called for all
 *                                  unknown events.
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetUnknownCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_Unknown unknownCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_Unknown_callback = unknownCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the BufferFlush event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param bufferFlushCallback      Function which should be called for all
 *                                  @eref{BufferFlush} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetBufferFlushCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_BufferFlush bufferFlushCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_BufferFlush_callback = bufferFlushCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MeasurementOnOff event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param measurementOnOffCallback Function which should be called for all
 *                                  @eref{MeasurementOnOff} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMeasurementOnOffCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MeasurementOnOff measurementOnOffCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MeasurementOnOff_callback = measurementOnOffCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the Enter event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param enterCallback            Function which should be called for all
 *                                  @eref{Enter} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetEnterCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                              OTF2_GlobalEvtReaderCallback_Enter enterCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_Enter_callback = enterCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the Leave event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param leaveCallback            Function which should be called for all
 *                                  @eref{Leave} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetLeaveCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                              OTF2_GlobalEvtReaderCallback_Leave leaveCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_Leave_callback = leaveCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiSend event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiSendCallback          Function which should be called for all
 *                                  @eref{MpiSend} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiSendCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_MpiSend mpiSendCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiSend_callback = mpiSendCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiIsend event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiIsendCallback         Function which should be called for all
 *                                  @eref{MpiIsend} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiIsend mpiIsendCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiIsend_callback = mpiIsendCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiIsendComplete event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiIsendCompleteCallback Function which should be called for all
 *                                  @eref{MpiIsendComplete} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiIsendCompleteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiIsendComplete mpiIsendCompleteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiIsendComplete_callback = mpiIsendCompleteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiIrecvRequest event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiIrecvRequestCallback  Function which should be called for all
 *                                  @eref{MpiIrecvRequest} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvRequestCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiIrecvRequest mpiIrecvRequestCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiIrecvRequest_callback = mpiIrecvRequestCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiRecv event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiRecvCallback          Function which should be called for all
 *                                  @eref{MpiRecv} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiRecvCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_MpiRecv mpiRecvCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiRecv_callback = mpiRecvCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiIrecv event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiIrecvCallback         Function which should be called for all
 *                                  @eref{MpiIrecv} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiIrecvCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiIrecv mpiIrecvCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiIrecv_callback = mpiIrecvCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiRequestTest event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiRequestTestCallback   Function which should be called for all
 *                                  @eref{MpiRequestTest} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiRequestTestCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiRequestTest mpiRequestTestCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiRequestTest_callback = mpiRequestTestCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiRequestCancelled event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param mpiRequestCancelledCallback Function which should be called for all
 *                                     @eref{MpiRequestCancelled}
 *                                     definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiRequestCancelledCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiRequestCancelled mpiRequestCancelledCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiRequestCancelled_callback = mpiRequestCancelledCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiCollectiveBegin event.
 *
 *  @param globalEvtReaderCallbacks   Struct for all callbacks.
 *  @param mpiCollectiveBeginCallback Function which should be called for all
 *                                    @eref{MpiCollectiveBegin} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiCollectiveBegin mpiCollectiveBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiCollectiveBegin_callback = mpiCollectiveBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the MpiCollectiveEnd event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param mpiCollectiveEndCallback Function which should be called for all
 *                                  @eref{MpiCollectiveEnd} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMpiCollectiveEndCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_MpiCollectiveEnd mpiCollectiveEndCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_MpiCollectiveEnd_callback = mpiCollectiveEndCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpFork event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompForkCallback          Function which should be called for all
 *                                  @eref{OmpFork} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpForkCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_OmpFork ompForkCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpFork_callback = ompForkCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpJoin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompJoinCallback          Function which should be called for all
 *                                  @eref{OmpJoin} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpJoinCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_OmpJoin ompJoinCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpJoin_callback = ompJoinCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpAcquireLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompAcquireLockCallback   Function which should be called for all
 *                                  @eref{OmpAcquireLock} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpAcquireLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_OmpAcquireLock ompAcquireLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpAcquireLock_callback = ompAcquireLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpReleaseLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompReleaseLockCallback   Function which should be called for all
 *                                  @eref{OmpReleaseLock} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpReleaseLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_OmpReleaseLock ompReleaseLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpReleaseLock_callback = ompReleaseLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpTaskCreate event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompTaskCreateCallback    Function which should be called for all
 *                                  @eref{OmpTaskCreate} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCreateCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_OmpTaskCreate ompTaskCreateCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpTaskCreate_callback = ompTaskCreateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpTaskSwitch event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompTaskSwitchCallback    Function which should be called for all
 *                                  @eref{OmpTaskSwitch} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpTaskSwitchCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_OmpTaskSwitch ompTaskSwitchCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpTaskSwitch_callback = ompTaskSwitchCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the OmpTaskComplete event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ompTaskCompleteCallback  Function which should be called for all
 *                                  @eref{OmpTaskComplete} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetOmpTaskCompleteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_OmpTaskComplete ompTaskCompleteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_OmpTaskComplete_callback = ompTaskCompleteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the Metric event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param metricCallback           Function which should be called for all
 *                                  @eref{Metric} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetMetricCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                               OTF2_GlobalEvtReaderCallback_Metric metricCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_Metric_callback = metricCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ParameterString event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param parameterStringCallback  Function which should be called for all
 *                                  @eref{ParameterString} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetParameterStringCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ParameterString parameterStringCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ParameterString_callback = parameterStringCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ParameterInt event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param parameterIntCallback     Function which should be called for all
 *                                  @eref{ParameterInt} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetParameterIntCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ParameterInt parameterIntCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ParameterInt_callback = parameterIntCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ParameterUnsignedInt event.
 *
 *  @param globalEvtReaderCallbacks     Struct for all callbacks.
 *  @param parameterUnsignedIntCallback Function which should be called for all
 *                                      @eref{ParameterUnsignedInt}
 *                                      definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetParameterUnsignedIntCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ParameterUnsignedInt parameterUnsignedIntCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ParameterUnsignedInt_callback = parameterUnsignedIntCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaWinCreate event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaWinCreateCallback     Function which should be called for all
 *                                  @eref{RmaWinCreate} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaWinCreateCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaWinCreate rmaWinCreateCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaWinCreate_callback = rmaWinCreateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaWinDestroy event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaWinDestroyCallback    Function which should be called for all
 *                                  @eref{RmaWinDestroy} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaWinDestroyCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaWinDestroy rmaWinDestroyCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaWinDestroy_callback = rmaWinDestroyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaCollectiveBegin event.
 *
 *  @param globalEvtReaderCallbacks   Struct for all callbacks.
 *  @param rmaCollectiveBeginCallback Function which should be called for all
 *                                    @eref{RmaCollectiveBegin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaCollectiveBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaCollectiveBegin rmaCollectiveBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaCollectiveBegin_callback = rmaCollectiveBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaCollectiveEnd event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaCollectiveEndCallback Function which should be called for all
 *                                  @eref{RmaCollectiveEnd} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaCollectiveEndCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaCollectiveEnd rmaCollectiveEndCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaCollectiveEnd_callback = rmaCollectiveEndCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaGroupSync event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaGroupSyncCallback     Function which should be called for all
 *                                  @eref{RmaGroupSync} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaGroupSyncCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaGroupSync rmaGroupSyncCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaGroupSync_callback = rmaGroupSyncCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaRequestLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaRequestLockCallback   Function which should be called for all
 *                                  @eref{RmaRequestLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaRequestLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaRequestLock rmaRequestLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaRequestLock_callback = rmaRequestLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaAcquireLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaAcquireLockCallback   Function which should be called for all
 *                                  @eref{RmaAcquireLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaAcquireLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaAcquireLock rmaAcquireLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaAcquireLock_callback = rmaAcquireLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaTryLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaTryLockCallback       Function which should be called for all
 *                                  @eref{RmaTryLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaTryLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaTryLock rmaTryLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaTryLock_callback = rmaTryLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaReleaseLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaReleaseLockCallback   Function which should be called for all
 *                                  @eref{RmaReleaseLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaReleaseLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaReleaseLock rmaReleaseLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaReleaseLock_callback = rmaReleaseLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaSync event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaSyncCallback          Function which should be called for all
 *                                  @eref{RmaSync} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaSyncCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                                OTF2_GlobalEvtReaderCallback_RmaSync rmaSyncCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaSync_callback = rmaSyncCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaWaitChange event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaWaitChangeCallback    Function which should be called for all
 *                                  @eref{RmaWaitChange} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaWaitChangeCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaWaitChange rmaWaitChangeCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaWaitChange_callback = rmaWaitChangeCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaPut event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaPutCallback           Function which should be called for all
 *                                  @eref{RmaPut} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaPutCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                               OTF2_GlobalEvtReaderCallback_RmaPut rmaPutCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaPut_callback = rmaPutCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaGet event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaGetCallback           Function which should be called for all
 *                                  @eref{RmaGet} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaGetCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                               OTF2_GlobalEvtReaderCallback_RmaGet rmaGetCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaGet_callback = rmaGetCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaAtomic event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaAtomicCallback        Function which should be called for all
 *                                  @eref{RmaAtomic} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaAtomicCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaAtomic rmaAtomicCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaAtomic_callback = rmaAtomicCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaOpCompleteBlocking event.
 *
 *  @param globalEvtReaderCallbacks      Struct for all callbacks.
 *  @param rmaOpCompleteBlockingCallback Function which should be called for all
 *                                       @eref{RmaOpCompleteBlocking}
 *                                       definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteBlockingCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaOpCompleteBlocking rmaOpCompleteBlockingCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaOpCompleteBlocking_callback = rmaOpCompleteBlockingCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaOpCompleteNonBlocking event.
 *
 *  @param globalEvtReaderCallbacks         Struct for all callbacks.
 *  @param rmaOpCompleteNonBlockingCallback Function which should be called for
 *                                          all
 *                                          @eref{RmaOpCompleteNonBlocking}
 *                                          definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteNonBlockingCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaOpCompleteNonBlocking rmaOpCompleteNonBlockingCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaOpCompleteNonBlocking_callback = rmaOpCompleteNonBlockingCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaOpTest event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param rmaOpTestCallback        Function which should be called for all
 *                                  @eref{RmaOpTest} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaOpTestCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaOpTest rmaOpTestCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaOpTest_callback = rmaOpTestCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the RmaOpCompleteRemote event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param rmaOpCompleteRemoteCallback Function which should be called for all
 *                                     @eref{RmaOpCompleteRemote}
 *                                     definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetRmaOpCompleteRemoteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_RmaOpCompleteRemote rmaOpCompleteRemoteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_RmaOpCompleteRemote_callback = rmaOpCompleteRemoteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadFork event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadForkCallback       Function which should be called for all
 *                                  @eref{ThreadFork} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadForkCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadFork threadForkCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadFork_callback = threadForkCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadJoin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadJoinCallback       Function which should be called for all
 *                                  @eref{ThreadJoin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadJoinCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadJoin threadJoinCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadJoin_callback = threadJoinCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadTeamBegin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadTeamBeginCallback  Function which should be called for all
 *                                  @eref{ThreadTeamBegin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadTeamBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadTeamBegin threadTeamBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadTeamBegin_callback = threadTeamBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadTeamEnd event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadTeamEndCallback    Function which should be called for all
 *                                  @eref{ThreadTeamEnd} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadTeamEndCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadTeamEnd threadTeamEndCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadTeamEnd_callback = threadTeamEndCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadAcquireLock event.
 *
 *  @param globalEvtReaderCallbacks  Struct for all callbacks.
 *  @param threadAcquireLockCallback Function which should be called for all
 *                                   @eref{ThreadAcquireLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadAcquireLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadAcquireLock threadAcquireLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadAcquireLock_callback = threadAcquireLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadReleaseLock event.
 *
 *  @param globalEvtReaderCallbacks  Struct for all callbacks.
 *  @param threadReleaseLockCallback Function which should be called for all
 *                                   @eref{ThreadReleaseLock} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadReleaseLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadReleaseLock threadReleaseLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadReleaseLock_callback = threadReleaseLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadTaskCreate event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadTaskCreateCallback Function which should be called for all
 *                                  @eref{ThreadTaskCreate} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCreateCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadTaskCreate threadTaskCreateCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadTaskCreate_callback = threadTaskCreateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadTaskSwitch event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadTaskSwitchCallback Function which should be called for all
 *                                  @eref{ThreadTaskSwitch} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadTaskSwitchCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadTaskSwitch threadTaskSwitchCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadTaskSwitch_callback = threadTaskSwitchCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadTaskComplete event.
 *
 *  @param globalEvtReaderCallbacks   Struct for all callbacks.
 *  @param threadTaskCompleteCallback Function which should be called for all
 *                                    @eref{ThreadTaskComplete} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadTaskCompleteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadTaskComplete threadTaskCompleteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadTaskComplete_callback = threadTaskCompleteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadCreate event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadCreateCallback     Function which should be called for all
 *                                  @eref{ThreadCreate} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadCreateCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadCreate threadCreateCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadCreate_callback = threadCreateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadBegin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadBeginCallback      Function which should be called for all
 *                                  @eref{ThreadBegin} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadBegin threadBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadBegin_callback = threadBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadWait event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadWaitCallback       Function which should be called for all
 *                                  @eref{ThreadWait} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadWaitCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadWait threadWaitCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadWait_callback = threadWaitCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ThreadEnd event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param threadEndCallback        Function which should be called for all
 *                                  @eref{ThreadEnd} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetThreadEndCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ThreadEnd threadEndCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ThreadEnd_callback = threadEndCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the CallingContextEnter event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param callingContextEnterCallback Function which should be called for all
 *                                     @eref{CallingContextEnter}
 *                                     definitions.
 *
 *  @since Version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetCallingContextEnterCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_CallingContextEnter callingContextEnterCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_CallingContextEnter_callback = callingContextEnterCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the CallingContextLeave event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param callingContextLeaveCallback Function which should be called for all
 *                                     @eref{CallingContextLeave}
 *                                     definitions.
 *
 *  @since Version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetCallingContextLeaveCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_CallingContextLeave callingContextLeaveCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_CallingContextLeave_callback = callingContextLeaveCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the CallingContextSample event.
 *
 *  @param globalEvtReaderCallbacks     Struct for all callbacks.
 *  @param callingContextSampleCallback Function which should be called for all
 *                                      @eref{CallingContextSample}
 *                                      definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetCallingContextSampleCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_CallingContextSample callingContextSampleCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_CallingContextSample_callback = callingContextSampleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoCreateHandle event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioCreateHandleCallback   Function which should be called for all
 *                                  @eref{IoCreateHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoCreateHandleCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoCreateHandle ioCreateHandleCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoCreateHandle_callback = ioCreateHandleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoDestroyHandle event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioDestroyHandleCallback  Function which should be called for all
 *                                  @eref{IoDestroyHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoDestroyHandleCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoDestroyHandle ioDestroyHandleCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoDestroyHandle_callback = ioDestroyHandleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoDuplicateHandle event.
 *
 *  @param globalEvtReaderCallbacks  Struct for all callbacks.
 *  @param ioDuplicateHandleCallback Function which should be called for all
 *                                   @eref{IoDuplicateHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoDuplicateHandleCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoDuplicateHandle ioDuplicateHandleCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoDuplicateHandle_callback = ioDuplicateHandleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoSeek event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioSeekCallback           Function which should be called for all
 *                                  @eref{IoSeek} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoSeekCallback(OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
                                                               OTF2_GlobalEvtReaderCallback_IoSeek ioSeekCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoSeek_callback = ioSeekCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoChangeStatusFlags event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param ioChangeStatusFlagsCallback Function which should be called for all
 *                                     @eref{IoChangeStatusFlags}
 *                                     definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoChangeStatusFlagsCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoChangeStatusFlags ioChangeStatusFlagsCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoChangeStatusFlags_callback = ioChangeStatusFlagsCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoDeleteFile event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioDeleteFileCallback     Function which should be called for all
 *                                  @eref{IoDeleteFile} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoDeleteFileCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoDeleteFile ioDeleteFileCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoDeleteFile_callback = ioDeleteFileCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoOperationBegin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioOperationBeginCallback Function which should be called for all
 *                                  @eref{IoOperationBegin} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoOperationBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoOperationBegin ioOperationBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoOperationBegin_callback = ioOperationBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoOperationTest event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioOperationTestCallback  Function which should be called for all
 *                                  @eref{IoOperationTest} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoOperationTestCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoOperationTest ioOperationTestCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoOperationTest_callback = ioOperationTestCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoOperationIssued event.
 *
 *  @param globalEvtReaderCallbacks  Struct for all callbacks.
 *  @param ioOperationIssuedCallback Function which should be called for all
 *                                   @eref{IoOperationIssued} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoOperationIssuedCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoOperationIssued ioOperationIssuedCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoOperationIssued_callback = ioOperationIssuedCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoOperationComplete event.
 *
 *  @param globalEvtReaderCallbacks    Struct for all callbacks.
 *  @param ioOperationCompleteCallback Function which should be called for all
 *                                     @eref{IoOperationComplete}
 *                                     definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoOperationCompleteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoOperationComplete ioOperationCompleteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoOperationComplete_callback = ioOperationCompleteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoOperationCancelled event.
 *
 *  @param globalEvtReaderCallbacks     Struct for all callbacks.
 *  @param ioOperationCancelledCallback Function which should be called for all
 *                                      @eref{IoOperationCancelled}
 *                                      definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoOperationCancelledCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoOperationCancelled ioOperationCancelledCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoOperationCancelled_callback = ioOperationCancelledCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoAcquireLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioAcquireLockCallback    Function which should be called for all
 *                                  @eref{IoAcquireLock} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoAcquireLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoAcquireLock ioAcquireLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoAcquireLock_callback = ioAcquireLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoReleaseLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioReleaseLockCallback    Function which should be called for all
 *                                  @eref{IoReleaseLock} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoReleaseLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoReleaseLock ioReleaseLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoReleaseLock_callback = ioReleaseLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the IoTryLock event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param ioTryLockCallback        Function which should be called for all
 *                                  @eref{IoTryLock} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetIoTryLockCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_IoTryLock ioTryLockCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_IoTryLock_callback = ioTryLockCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ProgramBegin event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param programBeginCallback     Function which should be called for all
 *                                  @eref{ProgramBegin} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetProgramBeginCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ProgramBegin programBeginCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ProgramBegin_callback = programBeginCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the ProgramEnd event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param programEndCallback       Function which should be called for all
 *                                  @eref{ProgramEnd} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetProgramEndCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_ProgramEnd programEndCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_ProgramEnd_callback = programEndCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the NonBlockingCollectiveRequest event.
 *
 *  @param globalEvtReaderCallbacks             Struct for all callbacks.
 *  @param nonBlockingCollectiveRequestCallback Function which should be called
 *                                              for all
 *                                              @eref{NonBlockingCollectiveRequest}
 *                                              definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveRequestCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_NonBlockingCollectiveRequest nonBlockingCollectiveRequestCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_NonBlockingCollectiveRequest_callback = nonBlockingCollectiveRequestCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the NonBlockingCollectiveComplete event.
 *
 *  @param globalEvtReaderCallbacks              Struct for all callbacks.
 *  @param nonBlockingCollectiveCompleteCallback Function which should be called
 *                                               for all
 *                                               @eref{NonBlockingCollectiveComplete}
 *                                               definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetNonBlockingCollectiveCompleteCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_NonBlockingCollectiveComplete nonBlockingCollectiveCompleteCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_NonBlockingCollectiveComplete_callback = nonBlockingCollectiveCompleteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the CommCreate event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param commCreateCallback       Function which should be called for all
 *                                  @eref{CommCreate} definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetCommCreateCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_CommCreate commCreateCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_CommCreate_callback = commCreateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the CommDestroy event.
 *
 *  @param globalEvtReaderCallbacks Struct for all callbacks.
 *  @param commDestroyCallback      Function which should be called for all
 *                                  @eref{CommDestroy} definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalEvtReaderCallbacks_SetCommDestroyCallback(
  OTF2_GlobalEvtReaderCallbacks* globalEvtReaderCallbacks,
  OTF2_GlobalEvtReaderCallback_CommDestroy commDestroyCallback) {
  globalEvtReaderCallbacks->OTF2_GlobalEvtReaderCallback_CommDestroy_callback = commDestroyCallback;
  return OTF2_SUCCESS;
}
