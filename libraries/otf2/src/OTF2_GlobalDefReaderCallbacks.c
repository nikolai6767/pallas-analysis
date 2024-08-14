#include <stdio.h>
#include <stdlib.h>

#include "pallas/pallas.h"
#include "otf2/OTF2_GlobalDefReaderCallbacks.h"
#include "otf2/otf2.h"

/** @brief Allocates a new struct for the global definition callbacks.
 *
 *  @return A newly allocated struct of type @eref{OTF2_GlobalDefReaderCallbacks}.
 */
OTF2_GlobalDefReaderCallbacks* OTF2_GlobalDefReaderCallbacks_New(void) {
  OTF2_GlobalDefReaderCallbacks* ret = malloc(sizeof(OTF2_GlobalDefReaderCallbacks));
  OTF2_GlobalDefReaderCallbacks_Clear(ret);
  return ret;
}

/** @brief Deallocates a struct for the global definition callbacks.
 *
 *  @param globalDefReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalDefReaderCallbacks_New}.
 */
void OTF2_GlobalDefReaderCallbacks_Delete(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks) {
  free(globalDefReaderCallbacks);
}

/** @brief Clears a struct for the global definition callbacks.
 *
 *  @param globalDefReaderCallbacks Handle to a struct previously allocated
 *                                  with @eref{OTF2_GlobalDefReaderCallbacks_New}.
 */
void OTF2_GlobalDefReaderCallbacks_Clear(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks) {
  memset(globalDefReaderCallbacks, 0, sizeof(OTF2_GlobalDefReaderCallbacks));
}

/** @brief Registers the callback for an unknown definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param unknownCallback          Function which should be called for all
 *                                  Unknown definitions.
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetUnknownCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                                OTF2_GlobalDefReaderCallback_Unknown unknownCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Unknown_callback = unknownCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{ClockProperties} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param clockPropertiesCallback  Function which should be called for all
 *                                  @eref{ClockProperties} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetClockPropertiesCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
									OTF2_GlobalDefReaderCallback_ClockProperties clockPropertiesCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_ClockProperties_callback = clockPropertiesCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{ParadigmProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param paradigmPropertyCallback Function which should be called for all
 *                                  @eref{ParadigmProperty} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetParadigmPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_ParadigmProperty paradigmPropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_ParadigmProperty_callback = paradigmPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoParadigm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioParadigmCallback       Function which should be called for all
 *                                  @eref{IoParadigm} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoParadigmCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoParadigm ioParadigmCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoParadigm_callback = ioParadigmCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{String} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param stringCallback           Function which should be called for all
 *                                  @eref{String} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetStringCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_String stringCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_String_callback = stringCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Attribute} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param attributeCallback        Function which should be called for all
 *                                  @eref{Attribute} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetAttributeCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Attribute attributeCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Attribute_callback = attributeCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNode} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param systemTreeNodeCallback   Function which should be called for all
 *                                  @eref{SystemTreeNode} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNode systemTreeNodeCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_SystemTreeNode_callback = systemTreeNodeCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationGroup} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param locationGroupCallback    Function which should be called for all
 *                                  @eref{LocationGroup} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationGroupCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationGroup locationGroupCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_LocationGroup_callback = locationGroupCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Location} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param locationCallback         Function which should be called for all
 *                                  @eref{Location} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Location locationCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Location_callback = locationCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Region} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param regionCallback           Function which should be called for all
 *                                  @eref{Region} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetRegionCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_Region regionCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Region_callback = regionCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Callsite} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callsiteCallback         Function which should be called for all
 *                                  @eref{Callsite} definitions.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallsiteCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Callsite callsiteCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Callsite_callback = callsiteCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Callpath} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callpathCallback         Function which should be called for all
 *                                  @eref{Callpath} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallpathCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Callpath callpathCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Callpath_callback = callpathCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Group} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param groupCallback            Function which should be called for all
 *                                  @eref{Group} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetGroupCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                              OTF2_GlobalDefReaderCallback_Group groupCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Group_callback = groupCallback;
						 return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricMember} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricMemberCallback     Function which should be called for all
 *                                  @eref{MetricMember} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricMemberCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricMember metricMemberCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_MetricMember_callback = metricMemberCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricClass} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricClassCallback      Function which should be called for all
 *                                  @eref{MetricClass} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricClassCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricClass metricClassCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_MetricClass_callback = metricClassCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricInstance} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param metricInstanceCallback   Function which should be called for all
 *                                  @eref{MetricInstance} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricInstanceCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricInstance metricInstanceCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_MetricInstance_callback = metricInstanceCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Comm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param commCallback             Function which should be called for all
 *                                  @eref{Comm} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCommCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                             OTF2_GlobalDefReaderCallback_Comm commCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Comm_callback = commCallback;
						return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{Parameter} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param parameterCallback        Function which should be called for all
 *                                  @eref{Parameter} definitions.
 *
 *  @since Version 1.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetParameterCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_Parameter parameterCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_Parameter_callback = parameterCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{RmaWin} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param rmaWinCallback           Function which should be called for all
 *                                  @eref{RmaWin} definitions.
 *
 *  @since Version 1.2
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetRmaWinCallback(OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
                                                               OTF2_GlobalDefReaderCallback_RmaWin rmaWinCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_RmaWin_callback = rmaWinCallback;
						  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{MetricClassRecorder} definition.
 *
 *  @param globalDefReaderCallbacks    Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetMetricClassRecorderCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_MetricClassRecorder metricClassRecorderCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_MetricClassRecorder_callback = metricClassRecorderCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNodeProperty} definition.
 *
 *  @param globalDefReaderCallbacks       Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodePropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty systemTreeNodePropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_SystemTreeNodeProperty_callback = systemTreeNodePropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SystemTreeNodeDomain} definition.
 *
 *  @param globalDefReaderCallbacks     Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSystemTreeNodeDomainCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain systemTreeNodeDomainCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_SystemTreeNodeDomain_callback = systemTreeNodeDomainCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationGroupProperty} definition.
 *
 *  @param globalDefReaderCallbacks      Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationGroupPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationGroupProperty locationGroupPropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_LocationGroupProperty_callback = locationGroupPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{LocationProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetLocationPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_LocationProperty locationPropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_LocationProperty_callback = locationPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartDimension} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartDimensionCallback    Function which should be called for all
 *                                  @eref{CartDimension} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartDimensionCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartDimension cartDimensionCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CartDimension_callback = cartDimensionCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartTopology} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartTopologyCallback     Function which should be called for all
 *                                  @eref{CartTopology} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartTopologyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartTopology cartTopologyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CartTopology_callback = cartTopologyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CartCoordinate} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param cartCoordinateCallback   Function which should be called for all
 *                                  @eref{CartCoordinate} definitions.
 *
 *  @since Version 1.3
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCartCoordinateCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CartCoordinate cartCoordinateCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CartCoordinate_callback = cartCoordinateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{SourceCodeLocation} definition.
 *
 *  @param globalDefReaderCallbacks   Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetSourceCodeLocationCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_SourceCodeLocation sourceCodeLocationCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_SourceCodeLocation_callback = sourceCodeLocationCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallingContext} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param callingContextCallback   Function which should be called for all
 *                                  @eref{CallingContext} definitions.
 *
 *  @since Version 1.5
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallingContextCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallingContext callingContextCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CallingContext_callback = callingContextCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallingContextProperty} definition.
 *
 *  @param globalDefReaderCallbacks       Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallingContextPropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallingContextProperty callingContextPropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CallingContextProperty_callback = callingContextPropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{InterruptGenerator} definition.
 *
 *  @param globalDefReaderCallbacks   Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetInterruptGeneratorCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_InterruptGenerator interruptGeneratorCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_InterruptGenerator_callback = interruptGeneratorCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoFileProperty} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioFilePropertyCallback   Function which should be called for all
 *                                  @eref{IoFileProperty} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoFilePropertyCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoFileProperty ioFilePropertyCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoFileProperty_callback = ioFilePropertyCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoRegularFile} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioRegularFileCallback    Function which should be called for all
 *                                  @eref{IoRegularFile} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoRegularFileCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoRegularFile ioRegularFileCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoRegularFile_callback = ioRegularFileCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoDirectory} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioDirectoryCallback      Function which should be called for all
 *                                  @eref{IoDirectory} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoDirectoryCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoDirectory ioDirectoryCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoDirectory_callback = ioDirectoryCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoHandle} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param ioHandleCallback         Function which should be called for all
 *                                  @eref{IoHandle} definitions.
 *
 *  @since Version 2.1
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoHandleCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoHandle ioHandleCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoHandle_callback = ioHandleCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{IoPreCreatedHandleState} definition.
 *
 *  @param globalDefReaderCallbacks        Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetIoPreCreatedHandleStateCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState ioPreCreatedHandleStateCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_IoPreCreatedHandleState_callback = ioPreCreatedHandleStateCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{CallpathParameter} definition.
 *
 *  @param globalDefReaderCallbacks  Struct for all callbacks.
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
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetCallpathParameterCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_CallpathParameter callpathParameterCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_CallpathParameter_callback = callpathParameterCallback;
  return OTF2_SUCCESS;
}

/** @brief Registers the callback for the @eref{InterComm} definition.
 *
 *  @param globalDefReaderCallbacks Struct for all callbacks.
 *  @param interCommCallback        Function which should be called for all
 *                                  @eref{InterComm} definitions.
 *
 *  @since Version 3.0
 *
 *  @retbegin
 *    @retcode{OTF2_SUCCESS, if successful}
 *    @retcode{OTF2_ERROR_INVALID_ARGUMENT,
 *             for an invalid @p defReaderCallbacks argument}
 *  @retend
 */
OTF2_ErrorCode OTF2_GlobalDefReaderCallbacks_SetInterCommCallback(
  OTF2_GlobalDefReaderCallbacks* globalDefReaderCallbacks,
  OTF2_GlobalDefReaderCallback_InterComm interCommCallback) {
  globalDefReaderCallbacks->OTF2_GlobalDefReaderCallback_InterComm_callback = interCommCallback;
  return OTF2_SUCCESS;
}
