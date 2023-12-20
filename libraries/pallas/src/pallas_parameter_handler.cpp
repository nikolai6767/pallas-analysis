/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include "pallas/pallas_parameter_handler.h"
#include <json/json.h>
#include <json/value.h>
#include <fstream>
#include <iostream>
#include "pallas/pallas_dbg.h"

std::string loadStringFromConfig(Json::Value& config, std::string fieldName) {
  if (config[fieldName]) {
    if (config[fieldName].isString()) {
      return config[fieldName].asString();
    }
  }
  return "";
}

uint64_t loadUInt64FromConfig(Json::Value& config, std::string fieldName) {
  if (config[fieldName]) {
    if (config[fieldName].isUInt64()) {
      return config[fieldName].asUInt64();
    }
  }
  return UINT64_MAX;
}

std::string loadStringFromEnv(Json::Value& config, std::string envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value)
    return std::string(env_value);
  return "";
}

uint64_t loadUInt64FromEnv(Json::Value& config, std::string envName) {
  const char* env_value = getenv(envName.c_str());
  if (env_value) {
    return std::stoull(env_value);
  }
  return UINT64_MAX;
}

std::map<pallas::CompressionAlgorithm, std::string> CompressionAlgorithmMap = {
  {pallas::CompressionAlgorithm::None, "None"},
  {pallas::CompressionAlgorithm::ZSTD, "ZSTD"},
  {pallas::CompressionAlgorithm::Histogram, "Histogram"},
#ifdef WITH_SZ
  {pallas::CompressionAlgorithm::SZ, "SZ"},
#endif
#ifdef WITH_ZFP
  {pallas::CompressionAlgorithm::ZFP, "ZFP"},
#endif
  {pallas::CompressionAlgorithm::Invalid, "Invalid"},
};

std::string pallas::toString(pallas::CompressionAlgorithm alg) {
  return CompressionAlgorithmMap[alg];
}

pallas::CompressionAlgorithm compressionAlgorithmFromString(std::string str) {
  for (auto& it : CompressionAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return pallas::CompressionAlgorithm::Invalid;
}

pallas::CompressionAlgorithm loadCompressionAlgorithmConfig(Json::Value& config) {
  pallas::CompressionAlgorithm ret = pallas::CompressionAlgorithm::None;

  std::string value = loadStringFromEnv(config, "PALLAS_COMPRESSION");
  if (value.empty()) {
    value = loadStringFromConfig(config, "compressionAlgorithm");
  }
  ret = compressionAlgorithmFromString(value);

  return ret;
}

std::map<pallas::EncodingAlgorithm, std::string> EncodingAlgorithmMap = {
  {pallas::EncodingAlgorithm::None, "None"},
  {pallas::EncodingAlgorithm::Masking, "Masking"},
  {pallas::EncodingAlgorithm::LeadingZeroes, "LeadingZeroes"},
  {pallas::EncodingAlgorithm::Invalid, "Invalid"},
};

std::string pallas::toString(pallas::EncodingAlgorithm alg) {
  return EncodingAlgorithmMap[alg];
}

pallas::EncodingAlgorithm encodingAlgorithmFromString(std::string str) {
  for (auto& it : EncodingAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return pallas::EncodingAlgorithm::Invalid;
}

pallas::EncodingAlgorithm loadEncodingAlgorithmConfig(Json::Value& config) {
  pallas::EncodingAlgorithm ret = pallas::EncodingAlgorithm::None;

  std::string value = loadStringFromEnv(config, "PALLAS_ENCODING");
  if (value.empty()) {
    value = loadStringFromConfig(config, "encodingAlgorithm");
  }
  ret = encodingAlgorithmFromString(value);
  return ret;
}

std::map<pallas::LoopFindingAlgorithm, std::string> LoopFindingAlgorithmMap = {
  {pallas::LoopFindingAlgorithm::None, "None"},
  {pallas::LoopFindingAlgorithm::Basic, "Basic"},
  {pallas::LoopFindingAlgorithm::BasicTruncated, "BasicTruncated"},
  {pallas::LoopFindingAlgorithm::Filter, "Filter"},
  {pallas::LoopFindingAlgorithm::Invalid, "Invalid"},
};

std::string pallas::toString(pallas::LoopFindingAlgorithm alg) {
  return LoopFindingAlgorithmMap[alg];
}

pallas::LoopFindingAlgorithm loopFindingAlgorithmFromString(std::string str) {
  for (auto& it : LoopFindingAlgorithmMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return pallas::LoopFindingAlgorithm::Invalid;
}

pallas::LoopFindingAlgorithm loadLoopFindingAlgorithmConfig(Json::Value& config) {
  pallas::LoopFindingAlgorithm ret = pallas::LoopFindingAlgorithm::BasicTruncated;

  std::string value = loadStringFromEnv(config, "PALLAS_LOOP_FINDING");
  if (value.empty()) {
    value = loadStringFromConfig(config, "loopFindingAlgorithm");
  }
  ret = loopFindingAlgorithmFromString(value);
  return ret;
}

uint64_t loadMaxLoopLength(Json::Value& config) {
  uint64_t ret = 100;

  uint64_t value = loadUInt64FromEnv(config, "PALLAS_LOOP_LENGTH");
  if (value == UINT64_MAX) {
    value = loadUInt64FromConfig(config, "maxLoopLength");
  }

  return value;
}

uint64_t loadZSTDCompressionLevel(Json::Value& config) {
  uint64_t ret = 3;

  uint64_t value = loadUInt64FromEnv(config, "PALLAS_ZSTD_LVL");
  if (value == UINT64_MAX) {
    value = loadUInt64FromConfig(config, "zstdCompressionLevel");
  }

  return value;
}

std::map<pallas::TimestampStorage, std::string> TimestampStorageMap = {{pallas::TimestampStorage::None, "None"},
                                                                    {pallas::TimestampStorage::Delta, "Delta"},
                                                                    {pallas::TimestampStorage::Timestamp, "Timestamp"},
                                                                    {pallas::TimestampStorage::Invalid, "Invalid"}};

std::string pallas::toString(pallas::TimestampStorage alg) {
  return TimestampStorageMap[alg];
}

pallas::TimestampStorage timestampStorageFromString(std::string str) {
  for (auto& it : TimestampStorageMap) {
    if (it.second.compare(str) == 0) {
      return it.first;
    }
  }
  return pallas::TimestampStorage::Invalid;
}

pallas::TimestampStorage loadTimestampStorageConfig(Json::Value& config) {
  pallas::TimestampStorage ret = pallas::TimestampStorage::Delta;

  std::string value = loadStringFromEnv(config, "PALLAS_TIMESTAMP_STORAGE");
  if (value.empty()) {
    value = loadStringFromConfig(config, "timestampStorage");
  }
  ret = timestampStorageFromString(value);
  return ret;
}

namespace pallas {
const char* defaultPath = "config.json";
const ParameterHandler parameterHandler = ParameterHandler();

ParameterHandler::ParameterHandler() {
  pallas_debug_level_init();

  std::ifstream configFile;
  const char* possibleConfigFileName = getenv("CONFIG_FILE_PATH");
  if (!possibleConfigFileName) {
    pallas_warn("No config file provided, using default: %s\n", defaultPath);
    possibleConfigFileName = defaultPath;
  }
  pallas_log(DebugLevel::Debug, "Loading configuration file from %s\n", possibleConfigFileName);
  configFile.open(possibleConfigFileName);
  if (!configFile.good()) {
    pallas_warn("Config file didn't exist: %s.\n", possibleConfigFileName);
    pallas_warn("Using the default one: %s\n", to_string().c_str());
    return;
  }

  Json::Value config;
  configFile >> config;
  configFile.close();
  /* Load from file */

  compressionAlgorithm = loadCompressionAlgorithmConfig(config);
  encodingAlgorithm = loadEncodingAlgorithmConfig(config);
  loopFindingAlgorithm = loadLoopFindingAlgorithmConfig(config);
  maxLoopLength = loadMaxLoopLength(config);
  zstdCompressionLevel = loadZSTDCompressionLevel(config);
  timestampStorage = loadTimestampStorageConfig(config);

  pallas_log(pallas::DebugLevel::Verbose, "%s\n", to_string().c_str());
}

size_t ParameterHandler::getMaxLoopLength() const {
  if (loopFindingAlgorithm == LoopFindingAlgorithm::BasicTruncated)
    return maxLoopLength;
  pallas_error("Asked for the max loop length but wasn't using a LoopFindingBasicTruncated algorithm.\n");
}
u_int8_t ParameterHandler::getZstdCompressionLevel() const {
  if (compressionAlgorithm == CompressionAlgorithm::ZSTD) {
    return zstdCompressionLevel;
  }
  pallas_error("Asked for ZSTD Compression Level but wasn't using a CompressionZSTD algorithm.\n");
}
CompressionAlgorithm ParameterHandler::getCompressionAlgorithm() const {
  return compressionAlgorithm;
}
EncodingAlgorithm ParameterHandler::getEncodingAlgorithm() const {
  if (isLossy(compressionAlgorithm) && encodingAlgorithm != EncodingAlgorithm::None) {
    pallas_warn("Encoding algorithm isn't None even though the compression algorithm is lossy.\n");
    return EncodingAlgorithm::None;
  }
  return encodingAlgorithm;
}
LoopFindingAlgorithm ParameterHandler::getLoopFindingAlgorithm() const {
  return loopFindingAlgorithm;
}

TimestampStorage ParameterHandler::getTimestampStorage() const {
  return timestampStorage;
}

std::string ParameterHandler::to_string() const {
  std::stringstream stream("");
  stream << "{\n";
  stream << '\t' << R"("compressionAlgorithm": ")" << toString(compressionAlgorithm) << "\",\n";
  stream << '\t' << R"("encodingAlgorithm": ")" << toString(encodingAlgorithm) << "\",\n";
  stream << '\t' << R"("loopFindingAlgorithm": ")" << toString(loopFindingAlgorithm) << "\",\n";
  stream << '\t' << R"("maxLoopLength": )" << maxLoopLength << ",\n";
  stream << '\t' << R"("zstdCompressionLevel": )" << zstdCompressionLevel << ",\n";
  stream << '\t' << R"("timestampStorage": ")" << toString(timestampStorage) << "\",\n";
  stream << "}";
  return stream.str();
}

}  // namespace pallas
