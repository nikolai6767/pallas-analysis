#include <stdio.h>
#include <stdlib.h>

#include "otf2/OTF2_DefReaderCallbacks.h"

OTF2_DefReaderCallbacks* OTF2_DefReaderCallbacks_New(void) {

  OTF2_DefReaderCallbacks* ret = malloc(sizeof(OTF2_DefReaderCallbacks));
  memset(ret, 0, sizeof(OTF2_DefReaderCallbacks));
  return ret;
}

/** @brief Deallocates a struct for the definition callbacks.
 *
 *  @param defReaderCallbacks Handle to a struct previously allocated
 *                            with @eref{OTF2_DefReaderCallbacks_New}.
 */
void OTF2_DefReaderCallbacks_Delete(OTF2_DefReaderCallbacks* defReaderCallbacks) {
  free(defReaderCallbacks);
}

/** @brief Clears a struct for the definition callbacks.
 *
 *  @param defReaderCallbacks Handle to a struct previously allocated
 *                            with @eref{OTF2_DefReaderCallbacks_New}.
 */
void OTF2_DefReaderCallbacks_Clear(OTF2_DefReaderCallbacks* defReaderCallbacks) {
  memset(defReaderCallbacks, 0, sizeof(OTF2_DefReaderCallbacks));
}

/** @brief Registers the callback for an unknown definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param unknownCallback    Function which should be called for all
 *                            unknown definitions.
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetUnknownCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                          OTF2_DefReaderCallback_Unknown unknownCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Unknown_callback = unknownCallback;
  return OTF2_SUCCESS;;
}

/** @brief Registers the callback for the @eref{MappingTable} definition.
 *
 *  @param defReaderCallbacks   Struct for all callbacks.
 *  @param mappingTableCallback Function which should be called for all
 *                              @eref{MappingTable} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetMappingTableCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_MappingTable mappingTableCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_MappingTable_callback =  mappingTableCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{ClockOffset} definition.
 *
 *  @param defReaderCallbacks  Struct for all callbacks.
 *  @param clockOffsetCallback Function which should be called for all
 *                             @eref{ClockOffset} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetClockOffsetCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                              OTF2_DefReaderCallback_ClockOffset clockOffsetCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_ClockOffset_callback = clockOffsetCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{String} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param stringCallback     Function which should be called for all
 *                            @eref{String} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetStringCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                         OTF2_DefReaderCallback_String stringCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_String_callback = stringCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Attribute} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param attributeCallback  Function which should be called for all
 *                            @eref{Attribute} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetAttributeCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                            OTF2_DefReaderCallback_Attribute attributeCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Attribute_callback = attributeCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNode} definition.
 *
 *  @param defReaderCallbacks     Struct for all callbacks.
 *  @param systemTreeNodeCallback Function which should be called for all
 *                                @eref{SystemTreeNode} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetSystemTreeNodeCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_SystemTreeNode systemTreeNodeCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_SystemTreeNode_callback = systemTreeNodeCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationGroup} definition.
 *
 *  @param defReaderCallbacks    Struct for all callbacks.
 *  @param locationGroupCallback Function which should be called for all
 *                               @eref{LocationGroup} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetLocationGroupCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_LocationGroup locationGroupCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_LocationGroup_callback = locationGroupCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Location} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param locationCallback   Function which should be called for all
 *                            @eref{Location} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetLocationCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                           OTF2_DefReaderCallback_Location locationCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Location_callback = locationCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Region} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param regionCallback     Function which should be called for all
 *                            @eref{Region} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetRegionCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                         OTF2_DefReaderCallback_Region regionCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Region_callback = regionCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Callsite} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param callsiteCallback   Function which should be called for all
 *                            @eref{Callsite} definitions.
 *
 *  @since Version 1.0
 *
 *  @deprecated In version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCallsiteCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                           OTF2_DefReaderCallback_Callsite callsiteCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Callsite_callback = callsiteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Callpath} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param callpathCallback   Function which should be called for all
 *                            @eref{Callpath} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCallpathCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                           OTF2_DefReaderCallback_Callpath callpathCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Callpath_callback = callpathCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Group} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param groupCallback      Function which should be called for all
 *                            @eref{Group} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetGroupCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                        OTF2_DefReaderCallback_Group groupCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Group_callback = groupCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricMember} definition.
 *
 *  @param defReaderCallbacks   Struct for all callbacks.
 *  @param metricMemberCallback Function which should be called for all
 *                              @eref{MetricMember} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetMetricMemberCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_MetricMember metricMemberCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_MetricMember_callback = metricMemberCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricClass} definition.
 *
 *  @param defReaderCallbacks  Struct for all callbacks.
 *  @param metricClassCallback Function which should be called for all
 *                             @eref{MetricClass} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetMetricClassCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                              OTF2_DefReaderCallback_MetricClass metricClassCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_MetricClass_callback = metricClassCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricInstance} definition.
 *
 *  @param defReaderCallbacks     Struct for all callbacks.
 *  @param metricInstanceCallback Function which should be called for all
 *                                @eref{MetricInstance} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetMetricInstanceCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_MetricInstance metricInstanceCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_MetricInstance_callback = metricInstanceCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Comm} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param commCallback       Function which should be called for all @eref{Comm}
 *                            definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCommCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                       OTF2_DefReaderCallback_Comm commCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Comm_callback = commCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Parameter} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param parameterCallback  Function which should be called for all
 *                            @eref{Parameter} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetParameterCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                            OTF2_DefReaderCallback_Parameter parameterCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_Parameter_callback = parameterCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{RmaWin} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param rmaWinCallback     Function which should be called for all
 *                            @eref{RmaWin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetRmaWinCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                         OTF2_DefReaderCallback_RmaWin rmaWinCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_RmaWin_callback = rmaWinCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricClassRecorder} definition.
 *
 *  @param defReaderCallbacks          Struct for all callbacks.
 *  @param metricClassRecorderCallback Function which should be called for all
 *                                     @eref{MetricClassRecorder}
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
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetMetricClassRecorderCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_MetricClassRecorder metricClassRecorderCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_MetricClassRecorder_callback = metricClassRecorderCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNodeProperty} definition.
 *
 *  @param defReaderCallbacks             Struct for all callbacks.
 *  @param systemTreeNodePropertyCallback Function which should be called for all
 *                                        @eref{SystemTreeNodeProperty}
 *                                        definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetSystemTreeNodePropertyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_SystemTreeNodeProperty systemTreeNodePropertyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_SystemTreeNodeProperty_callback = systemTreeNodePropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNodeDomain} definition.
 *
 *  @param defReaderCallbacks           Struct for all callbacks.
 *  @param systemTreeNodeDomainCallback Function which should be called for all
 *                                      @eref{SystemTreeNodeDomain}
 *                                      definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetSystemTreeNodeDomainCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_SystemTreeNodeDomain systemTreeNodeDomainCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_SystemTreeNodeDomain_callback = systemTreeNodeDomainCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationGroupProperty} definition.
 *
 *  @param defReaderCallbacks            Struct for all callbacks.
 *  @param locationGroupPropertyCallback Function which should be called for all
 *                                       @eref{LocationGroupProperty}
 *                                       definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetLocationGroupPropertyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_LocationGroupProperty locationGroupPropertyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_LocationGroupProperty_callback = locationGroupPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationProperty} definition.
 *
 *  @param defReaderCallbacks       Struct for all callbacks.
 *  @param locationPropertyCallback Function which should be called for all
 *                                  @eref{LocationProperty} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetLocationPropertyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_LocationProperty locationPropertyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_LocationProperty_callback = locationPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartDimension} definition.
 *
 *  @param defReaderCallbacks    Struct for all callbacks.
 *  @param cartDimensionCallback Function which should be called for all
 *                               @eref{CartDimension} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCartDimensionCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CartDimension cartDimensionCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CartDimension_callback = cartDimensionCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartTopology} definition.
 *
 *  @param defReaderCallbacks   Struct for all callbacks.
 *  @param cartTopologyCallback Function which should be called for all
 *                              @eref{CartTopology} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCartTopologyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CartTopology cartTopologyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CartTopology_callback = cartTopologyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartCoordinate} definition.
 *
 *  @param defReaderCallbacks     Struct for all callbacks.
 *  @param cartCoordinateCallback Function which should be called for all
 *                                @eref{CartCoordinate} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCartCoordinateCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CartCoordinate cartCoordinateCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CartCoordinate_callback = cartCoordinateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SourceCodeLocation} definition.
 *
 *  @param defReaderCallbacks         Struct for all callbacks.
 *  @param sourceCodeLocationCallback Function which should be called for all
 *                                    @eref{SourceCodeLocation} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetSourceCodeLocationCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_SourceCodeLocation sourceCodeLocationCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_SourceCodeLocation_callback = sourceCodeLocationCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallingContext} definition.
 *
 *  @param defReaderCallbacks     Struct for all callbacks.
 *  @param callingContextCallback Function which should be called for all
 *                                @eref{CallingContext} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCallingContextCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CallingContext callingContextCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CallingContext_callback = callingContextCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallingContextProperty} definition.
 *
 *  @param defReaderCallbacks             Struct for all callbacks.
 *  @param callingContextPropertyCallback Function which should be called for all
 *                                        @eref{CallingContextProperty}
 *                                        definitions.
 *
 *  @since Version 2.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCallingContextPropertyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CallingContextProperty callingContextPropertyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CallingContextProperty_callback = callingContextPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{InterruptGenerator} definition.
 *
 *  @param defReaderCallbacks         Struct for all callbacks.
 *  @param interruptGeneratorCallback Function which should be called for all
 *                                    @eref{InterruptGenerator} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetInterruptGeneratorCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_InterruptGenerator interruptGeneratorCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_InterruptGenerator_callback = interruptGeneratorCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoFileProperty} definition.
 *
 *  @param defReaderCallbacks     Struct for all callbacks.
 *  @param ioFilePropertyCallback Function which should be called for all
 *                                @eref{IoFileProperty} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetIoFilePropertyCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_IoFileProperty ioFilePropertyCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_IoFileProperty_callback = ioFilePropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoRegularFile} definition.
 *
 *  @param defReaderCallbacks    Struct for all callbacks.
 *  @param ioRegularFileCallback Function which should be called for all
 *                               @eref{IoRegularFile} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetIoRegularFileCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_IoRegularFile ioRegularFileCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_IoRegularFile_callback = ioRegularFileCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoDirectory} definition.
 *
 *  @param defReaderCallbacks  Struct for all callbacks.
 *  @param ioDirectoryCallback Function which should be called for all
 *                             @eref{IoDirectory} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetIoDirectoryCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                              OTF2_DefReaderCallback_IoDirectory ioDirectoryCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_IoDirectory_callback = ioDirectoryCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoHandle} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param ioHandleCallback   Function which should be called for all
 *                            @eref{IoHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetIoHandleCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                           OTF2_DefReaderCallback_IoHandle ioHandleCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_IoHandle_callback = ioHandleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoPreCreatedHandleState} definition.
 *
 *  @param defReaderCallbacks              Struct for all callbacks.
 *  @param ioPreCreatedHandleStateCallback Function which should be called for
 *                                         all @eref{IoPreCreatedHandleState}
 *                                         definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetIoPreCreatedHandleStateCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_IoPreCreatedHandleState ioPreCreatedHandleStateCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_IoPreCreatedHandleState_callback = ioPreCreatedHandleStateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallpathParameter} definition.
 *
 *  @param defReaderCallbacks        Struct for all callbacks.
 *  @param callpathParameterCallback Function which should be called for all
 *                                   @eref{CallpathParameter} definitions.
 *
 *  @since Version 2.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetCallpathParameterCallback(
  OTF2_DefReaderCallbacks* defReaderCallbacks,
  OTF2_DefReaderCallback_CallpathParameter callpathParameterCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_CallpathParameter_callback = callpathParameterCallback;
  return OTF2_SUCCESS;
}

/** @brief Regiseters the callback for the @eref{InterComm} definition.
 *
 *  @param defReaderCallbacks Struct for all callbacks.
 *  @param interCommCallback  Function which should be called for all
 *                            @eref{InterComm} definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_DefReaderCallbacks_SetInterCommCallback(OTF2_DefReaderCallbacks* defReaderCallbacks,
                                                            OTF2_DefReaderCallback_InterComm interCommCallback) {
  defReaderCallbacks->OTF2_DefReaderCallback_InterComm_callback = interCommCallback;
  return OTF2_SUCCESS;
}
