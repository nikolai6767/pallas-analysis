/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */

#include <libgen.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#ifdef WITH_ZFP
#include <zfp.h>
#endif
#ifdef WITH_SZ
#include <sz.h>
#endif

#include "pallas/pallas.h"
#include "pallas/pallas_dbg.h"
#include "pallas/pallas_hash.h"
#include "pallas/pallas_parameter_handler.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"

short STORE_TIMESTAMPS = 1;
static short STORE_HASHING = 0;
void pallas_storage_option_init() {
  // Timestamp storage
  const char* store_timestamps_str = getenv("STORE_TIMESTAMPS");
  if (store_timestamps_str && strcmp(store_timestamps_str, "TRUE") != 0)
    STORE_TIMESTAMPS = 0;

  // Store hash for sequences
  const char* store_hashing_str = getenv("STORE_HASHING");
  if (store_hashing_str && strcmp(store_hashing_str, "FALSE") != 0)
    STORE_HASHING = 1;
}

static void pallasStoreEvent(pallas::EventSummary& event, FILE* eventFile, FILE* durationFile);
static void pallasStoreSequence(pallas::Sequence& sequence, FILE* sequenceFile, FILE* durationFile);

static void pallasStoreLoop(pallas::Loop& loop, FILE* loopFile);

static void pallasStoreString(pallas::String* s, int string_index, FILE* stringFile);
static void pallasStoreRegions(pallas::Archive* a);
static void pallasStoreAttributes(pallas::Archive* a);

static void pallasStoreLocationGroups(pallas::Archive* a);
static void pallasStoreLocations(pallas::Archive* a);

static void pallasReadEvent(pallas::EventSummary& event,
                            FILE* eventFile,
                            FILE* durationFile,
                            const char* durationFileName);
static void pallasReadSequence(pallas::Sequence& sequence,
                               FILE* sequenceFile,
                               FILE* durationFile,
                               const char* durationFileName);
static void pallasReadLoop(pallas::Loop& loop, FILE* loopFile);

static void pallasReadString(pallas::String* s, int string_index, FILE* stringFile);
static void pallasReadRegions(pallas::Archive* a);
static void pallasReadAttributes(pallas::Archive* a);
static void pallasReadLocationGroups(pallas::Archive* a);
static void pallasReadLocations(pallas::Archive* a);

void pallasLoadThread(pallas::Archive* archive, pallas::ThreadId thread_id);

static pallas::Archive* pallasGetArchive(pallas::Archive* global_archive,
                                         pallas::LocationGroupId archive_id,
                                         bool print_warning = true);

static int _mkdir(const char *dir, __mode_t mode) {
  char tmp[1024];
  char *p = NULL;
  size_t len;

  snprintf(tmp, sizeof(tmp),"%s",dir);
  len = strlen(tmp);
  if (tmp[len - 1] == '/')
    tmp[len - 1] = 0;
  for (p = tmp + 1; *p; p++)
    if (*p == '/') {
      *p = 0;
      mkdir(tmp, mode);
      *p = '/';
    }
  return mkdir(tmp, mode);
}

static void pallasMkdir(const char* dirname, mode_t mode) {
  if (_mkdir(dirname, mode) != 0) {
    if (errno != EEXIST)
      pallas_error("mkdir(%s) failed: %s\n", dirname, strerror(errno));
  }
}

static FILE* pallasFileOpen(const char* filename, const char* mode) {
  pallas_log(pallas::DebugLevel::Debug, "Open %s with mode %s\n", filename, mode);
  char* filename_copy = strdup(filename);
  pallasMkdir(dirname(filename_copy), 0777);
  free(filename_copy);

  FILE* file = fopen(filename, mode);
  if (file == nullptr) {
    pallas_error("Cannot open %s: %s\n", filename, strerror(errno));
  }
  return file;
}

#define _pallas_fread(ptr, size, nmemb, stream)            \
  do {                                                     \
    size_t ret = fread(ptr, size, nmemb, stream);          \
    if (ret != nmemb)                                      \
      pallas_error("fread failed: %s\n", strerror(errno)); \
  } while (0)

#define _pallas_fwrite(ptr, size, nmemb, stream)   \
  do {                                             \
    size_t ret = fwrite(ptr, size, nmemb, stream); \
    if (ret != nmemb)                              \
      pallas_error("fwrite failed\n");             \
  } while (0)

/******************* Read/Write/Compression function for vectors and arrays *******************/

/** Compresses the content in src using ZSTD and writes it to dest. Returns the amount of data written.
 *  @param src The source array.
 *  @param size Size of the source array.
 *  @param dest A free array in which the compressed data will be written.
 *  @param destSize Size of the destination array
 *  @returns Number of bytes written in the dest array.
 */
inline static size_t _pallas_zstd_compress(void* src, size_t size, void* dest, size_t destSize) {
  return ZSTD_compress(dest, destSize, src, size, pallas::parameterHandler->getZstdCompressionLevel());
}

/**
 * Decompresses an array that has been compressed by ZSTD. Returns the size of the uncompressed data.
 * @param realSize Size of the uncompressed data.
 * @param compArray The compressed array.
 * @param compSize Size of the compressed array.
 * @returns Uncompressed array.
 */
inline static uint64_t* _pallas_zstd_read(size_t& realSize, void* compArray, size_t compSize) {
  realSize = ZSTD_getFrameContentSize(compArray, compSize);
  auto dest = new byte[realSize];
  ZSTD_decompress(dest, realSize, compArray, compSize);
  return reinterpret_cast<uint64_t*>(dest);
}

#ifdef WITH_ZFP
/**
 * Gives a conservative upper bound for the size of the compressed data.
 * @param src The source array.
 * @param n Number of items in the array.
 * @return Upper bound to compressed array size in bytes.
 */
inline static size_t _pallas_zfp_bound(uint64_t* src, size_t n) {
  zfp_type type = zfp_type_int64;                 // array scalar type
  zfp_field* field = zfp_field_1d(src, type, n);  // array metadata
  zfp_stream* zfp = zfp_stream_open(nullptr);     // compressed stream and parameters
  zfp_stream_set_accuracy(zfp, .1);               // set tolerance for fixed-accuracy mode, this is absolute error
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  zfp_stream_close(zfp);
  return bufsize;
}
/**
 * Compresses the content in src using the 1D ZFP Algorithm and writes it to dest.
 * Returns the amounts of data written.
 * @param src The source array.
 * @param n Number of items in the source array.
 * @param dest A free array in which the compressed data will be written.
 * @param destSize Size of the destination array.
 * @return Number of bytes written in the dest array.
 */
inline static size_t _pallas_zfp_compress(uint64_t* src, size_t n, void* dest, size_t destSize) {
  zfp_type type = zfp_type_int64;                 // array scalar type
  zfp_field* field = zfp_field_1d(src, type, n);  // array metadata
  zfp_stream* zfp = zfp_stream_open(nullptr);     // compressed stream and parameters
  zfp_stream_set_accuracy(zfp, .1);               // set tolerance for fixed-accuracy mode, this is absolute error
  size_t bufsize = zfp_stream_maximum_size(zfp, field);  // capacity of compressed buffer (conservative)
  pallas_assert(bufsize <= destSize);
  bitstream* stream = stream_open(dest, bufsize);  // bit stream to compress to
  zfp_stream_set_bit_stream(zfp, stream);          // associate with compressed stream
  zfp_stream_rewind(zfp);                          // rewind stream to beginning
  size_t outSize = zfp_compress(zfp, field);       // return value is byte size of compressed stream
  zfp_stream_close(zfp);
  stream_close(stream);
  return outSize;
}

/**
 * Decompresses the content in src using the 1D ZFP Algorithm and writes it to dest.
 * Returns the amounts of data written.
 * @param n Number of items that should be decompressed.
 * @param compressedArray The compressed array.
 * @param destSize Size of the compressed array.
 * @returns Uncompressed array of size uint64 * n.
 */
inline static uint64_t* _pallas_zfp_decompress(size_t n, void* compressedArray, size_t compressedSize) {
  auto dest = new uint64_t[n];
  zfp_type type = zfp_type_int64;                  // array scalar type
  zfp_field* field = zfp_field_1d(dest, type, n);  // array metadata
  zfp_stream* zfp = zfp_stream_open(nullptr);      // compressed stream and parameters
  zfp_stream_set_accuracy(zfp, .1);                // set tolerance for fixed-accuracy mode, this is absolute error
  bitstream* stream = stream_open(compressedArray, compressedSize);  // bit stream to read from
  zfp_stream_set_bit_stream(zfp, stream);                            // associate with compressed stream
  zfp_stream_rewind(zfp);                                            // rewind stream to beginning
  size_t outSize = zfp_decompress(zfp, field);                       // return value is byte size of compressed stream
  zfp_stream_close(zfp);
  stream_close(stream);
  return dest;
}
#endif
#ifdef WITH_SZ
/**
 * Compresses the content in src using the 1D SZ Algorithm.
 * @param src The source array.
 * @param n Number of items in the source array.
 * @param compressedSize Size of the compressed array. Passed by ref and modified.
 * @return The compressed array.
 */
inline static byte* _pallas_sz_compress(uint64_t* src, size_t n, size_t& compressedSize) {
  SZ_Init(nullptr);
  byte* compressedArray = SZ_compress(SZ_UINT64, src, &compressedSize, 0, 0, 0, 0, n);
  SZ_Finalize();
  return compressedArray;
}

inline static uint64_t* _pallas_sz_decompress(size_t n, byte* compressedArray, size_t compressedSize) {
  return static_cast<uint64_t*>(SZ_decompress(SZ_UINT64, compressedArray, compressedSize, 0, 0, 0, 0, n));
};

#endif

#define N_BYTES 1
#define N_BITS (N_BYTES * 8)
#define MAX_BIT ((1 << N_BITS) - 1)

/** Compresses the content in src using the Histogram method and writes it to dest.
 * Returns the amount of data written.
 *  @param src The source array.
 *  @param n Number of elements in src.
 *  @param dest A free array in which the compressed data will be written.
 *  @param destSize Size of the destination array
 *  @returns Number of bytes written in the dest array.
 */
inline static size_t _pallas_histogram_compress(const uint64_t* src, size_t n, byte* dest, size_t destSize) {
  pallas_assert(destSize >= (N_BYTES * n + 2 * sizeof(uint64_t)));
  // Compute the min and max
  uint64_t min = UINT64_MAX, max = 0;
  for (size_t i = 0; i < n; i++) {
    min = (src[i] < min) ? src[i] : min;
    max = (src[i] > max) ? src[i] : max;
  }
  size_t width = max - min;
  size_t stepSize = ((double)width) / MAX_BIT;
  printf("Min: %lu; Max: %lu\n", min, max);
  // TODO Check size is sufficient
  memcpy(dest, &min, sizeof(min));
  dest = &dest[sizeof(min)];
  memcpy(dest, &max, sizeof(max));
  dest = &dest[sizeof(max)];
  for (size_t i = 0; i < n; i++) {
    auto temp = (size_t)std::round((src[i] - min) / stepSize);
    temp = (temp > MAX_BIT) ? MAX_BIT : temp;
    printf("Writing %lu as %lu\n", src[i], temp);
    memcpy(&dest[i * N_BYTES], &temp, N_BYTES);
  }

  return N_BYTES * n + 2 * sizeof(uint64_t);
}

/** Decompresses the content in compArray using the Histogram method and writes it to dest.
 * Returns the amount of data written.
 * @param n Number of elements in the dest array.
 * @param compArray The compressed array.
 * @param compSize Size of the compressed array.
 * @returns Array of uncompressed data of size uint64_t * n.
 */
inline static uint64_t* _pallas_histogram_read(size_t n, byte* compArray, size_t compSize) {
  auto dest = new uint64_t[n];
  // Compute the min and max
  uint64_t min, max;
  memcpy(&min, compArray, sizeof(min));
  compArray = &compArray[sizeof(min)];
  memcpy(&max, compArray, sizeof(max));
  compArray = &compArray[sizeof(max)];
  size_t width = max - min;
  size_t stepSize = width / (1 << N_BITS);

  printf("Min: %lu; Max: %lu\n", min, max);

  for (size_t i = 0; i < n; i++) {
    size_t factor = 0;
    memcpy(&factor, &compArray[i * N_BYTES], N_BYTES);
    dest[i] = min + factor * stepSize;
    printf("Reading %lu as %lu\n", factor, dest[i]);
  }
  return dest;
}

/**
 * Encodes the content in src using a Masking technique and writes it to dest.
 * This is done only for 64-bits values.
 * @param src The source array. Contains n elements of 8 bytes (sizeof uint64_t).
 * @param dest The destination array. Same size as src, is an uint8 for convenience (byte counting).
 * @param n Number of elements in source array.
 * @return Number of interesting bytes contained in dest. (0 <= nBytes <= n * sizeof uint64)
 */
inline static size_t _pallas_masking_encode(const uint64_t* src, byte* dest, size_t n) {
  uint64_t mask = 0;
  for (int i = 0; i < n; i++) {
    mask |= src[i];
  }
  short maskSize = 0;
  while (mask != 0) {
    mask >>= 8;
    maskSize += 1;
  }
  // maskSize is the number of bytes needed to write the mask
  // ie the most amount of byte any number in src will need to be written
  if (maskSize && maskSize != sizeof(uint64_t)) {
    for (int i = 0; i < n; i++) {
      // FIXME This works because our LSB is in front (Small-endian)
      memcpy(&dest[maskSize * i], &src[i], maskSize);
    }
    return maskSize * n;
  } else {
    memcpy(dest, src, n * sizeof(uint64_t));
    return n * sizeof(uint64_t);
  }
}

/** De-encodes an array that has been compressed by the Masking technique. Returns the size of the unencoded data.
 * @param n Number of elements in the dest array.
 * @param encodedArray The encoded array.
 * @param encodedSize Size of the encoded array.
 * @returns Decoded array.
 */
inline static uint64_t* _pallas_masking_read(size_t n, byte* encodedArray, size_t encodedSize) {
  auto dest = new uint64_t[n];
  size_t size = n * sizeof(uint64_t);
  if (encodedSize == size) {
    memcpy(dest, encodedArray, size);
    return dest;
  }
  size_t width = encodedSize / n;
  // width is the number of bytes needed to write an element in the encoded array.
  memset(dest, 0, size);
  for (int i = 0; i < n; i++) {
    // FIXME Still only works with Little-Endian architecture.
    memcpy(&dest[i], &encodedArray[width * i], width);
  }
  return dest;
}

size_t numberRawBytes = 0;
size_t numberCompressedBytes = 0;

/**
 * Writes the array to the given file, but encodes and compresses it before
 * according to the value of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param src The source array. Contains n elements of 8 bytes (sizeof uint64_t).
 * @param n Number of elements in src.
 * @param file File to write in.
 */
inline static void _pallas_compress_write(uint64_t* src, size_t n, FILE* file) {
  size_t size = n * sizeof(uint64_t);
  uint64_t* encodedArray = nullptr;
  size_t encodedSize;
  // First we do the encoding
  switch (pallas::parameterHandler->getEncodingAlgorithm()) {
  case pallas::EncodingAlgorithm::None:
    break;
  case pallas::EncodingAlgorithm::Masking: {
    encodedArray = new uint64_t[n];
    encodedSize = _pallas_masking_encode(src, (uint8_t*)encodedArray, n);
    break;
  }
  case pallas::EncodingAlgorithm::LeadingZeroes: {
    pallas_error("Not yet implemented\n");
    break;
  }
  default:
    pallas_error("Invalid Encoding algorithm\n");
  }

  byte* compressedArray = nullptr;
  size_t compressedSize;
  switch (pallas::parameterHandler->getCompressionAlgorithm()) {
  case pallas::CompressionAlgorithm::None:
    break;
  case pallas::CompressionAlgorithm::ZSTD: {
    compressedSize = ZSTD_compressBound(encodedArray ? encodedSize : size);
    compressedArray = new byte[compressedSize];
    if (encodedArray) {
      compressedSize = _pallas_zstd_compress(encodedArray, encodedSize, compressedArray, compressedSize);
    } else {
      compressedSize = _pallas_zstd_compress(src, size, compressedArray, compressedSize);
    }
    break;
  }
  case pallas::CompressionAlgorithm::Histogram: {
    compressedSize = (n + 2) * sizeof(uint64_t);  // Take into account that we add the min and the max.
    compressedArray = new uint8_t[compressedSize];
    compressedSize = _pallas_histogram_compress(src, n, compressedArray, compressedSize);
    break;
  }
#ifdef WITH_ZFP
  case pallas::CompressionAlgorithm::ZFP:
    compressedSize = _pallas_zfp_bound(src, n);
    compressedArray = new byte[compressedSize];
    compressedSize = _pallas_zfp_compress(src, n, compressedArray, compressedSize);
    break;
#endif
#ifdef WITH_SZ
  case pallas::CompressionAlgorithm::SZ:
    compressedArray = _pallas_sz_compress(src, n, compressedSize);
    break;
#endif
  default:
    pallas_error("Invalid Compression algorithm\n");
  }

  if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
    pallas_log(pallas::DebugLevel::Debug, "Compressing %lu bytes as %lu bytes\n", size, compressedSize);
    _pallas_fwrite(&compressedSize, sizeof(compressedSize), 1, file);
    _pallas_fwrite(compressedArray, compressedSize, 1, file);
    numberRawBytes += size;
    numberCompressedBytes += compressedSize;
  } else if (pallas::parameterHandler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None) {
    pallas_log(pallas::DebugLevel::Debug, "Encoding %lu bytes as %lu bytes\n", size, encodedSize);
    _pallas_fwrite(&encodedSize, sizeof(encodedSize), 1, file);
    _pallas_fwrite(encodedArray, encodedSize, 1, file);
  } else {
    pallas_log(pallas::DebugLevel::Debug, "Writing %lu bytes as is.\n", size);
    _pallas_fwrite(&size, sizeof(size), 1, file);
    _pallas_fwrite(src, size, 1, file);
  }
  if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None)
    delete[] compressedArray;
  if (pallas::parameterHandler->getEncodingAlgorithm() != pallas::EncodingAlgorithm::None)
    delete[] encodedArray;
}

/**
 * Reads, de-encodes and decompresses an array from the given file,
 * according to the values of parameterHandler::EncodingAlgorithm and parameterHandler::CompressingAlgorithm.
 * @param n Number of elements of 8 bytes dest is supposed to have.
 * @param file File to read from
 * @returns Array of uncompressed data of size uint64_t * n.
 */
inline static uint64_t* _pallas_compress_read(size_t n, FILE* file) {
  size_t expectedSize = n * sizeof(uint64_t);
  uint64_t* uncompressedArray = nullptr;

  size_t compressedSize;
  byte* compressedArray = nullptr;

  size_t encodedSize;
  byte* encodedArray = nullptr;

  auto compressionAlgorithm = pallas::parameterHandler->getCompressionAlgorithm();
  auto encodingAlgorithm = pallas::parameterHandler->getEncodingAlgorithm();
  if (compressionAlgorithm != pallas::CompressionAlgorithm::None) {
    _pallas_fread(&compressedSize, sizeof(compressedSize), 1, file);
    compressedArray = new byte[compressedSize];
    _pallas_fread(compressedArray, compressedSize, 1, file);
  }

  switch (compressionAlgorithm) {
  case pallas::CompressionAlgorithm::None:
    break;
  case pallas::CompressionAlgorithm::ZSTD: {
    if (pallas::parameterHandler->getEncodingAlgorithm() == pallas::EncodingAlgorithm::None) {
      size_t uncompressedSize;
      uncompressedArray = _pallas_zstd_read(uncompressedSize, compressedArray, compressedSize);
      pallas_assert(uncompressedSize == expectedSize);
    } else {
      encodedArray = reinterpret_cast<byte*>(_pallas_zstd_read(encodedSize, compressedArray, compressedSize));
      pallas_assert(encodedSize <= expectedSize);
    }
    delete[] compressedArray;
    break;
  }
  case pallas::CompressionAlgorithm::Histogram: {
    uncompressedArray = _pallas_histogram_read(n, compressedArray, compressedSize);
    break;
  }
#ifdef WITH_ZFP
  case pallas::CompressionAlgorithm::ZFP: {
    uncompressedArray = _pallas_zfp_decompress(n, compressedArray, compressedSize);
    break;
  }
#endif
#ifdef WITH_SZ
  case pallas::CompressionAlgorithm::SZ:
    uncompressedArray = _pallas_sz_decompress(n, compressedArray, compressedSize);
    break;
#endif
  default:
    pallas_error("Invalid Compression algorithm\n");
  }

  switch (encodingAlgorithm) {
  case pallas::EncodingAlgorithm::None:
    break;
  case pallas::EncodingAlgorithm::Masking: {
    if (pallas::parameterHandler->getCompressionAlgorithm() == pallas::CompressionAlgorithm::None) {
      _pallas_fread(&encodedSize, sizeof(encodedSize), 1, file);
      encodedArray = new byte[encodedSize];  // Too big but don't care
      _pallas_fread(encodedArray, encodedSize, 1, file);
    }
    uncompressedArray = _pallas_masking_read(n, encodedArray, encodedSize);
    delete[] encodedArray;
    break;
  }
  case pallas::EncodingAlgorithm::LeadingZeroes: {
    pallas_error("Not yet implemented\n");
    break;
  }
  default:
    pallas_error("Invalid Encoding algorithm\n");
  }

  if (compressionAlgorithm == pallas::CompressionAlgorithm::None &&
      encodingAlgorithm == pallas::EncodingAlgorithm::None) {
    size_t realSize;
    _pallas_fread(&realSize, sizeof(realSize), 1, file);
    uncompressedArray = new uint64_t[n];
    _pallas_fread(uncompressedArray, realSize, 1, file);
    pallas_assert(realSize == n * sizeof(uint64_t));
  }
  return uncompressedArray;
}

void pallas::LinkedVector::writeToFile(FILE* vectorFile, FILE* valueFile) {
  finalUpdateStats();
  // Write the statistics to the vectorFile
  _pallas_fwrite(&size, sizeof(size), 1, vectorFile);
  if (size == 1) {
    _pallas_fwrite(&min, sizeof(min), 1, vectorFile);
  }
  if (size >= 2) {
    _pallas_fwrite(&min, sizeof(min), 1, vectorFile);
    _pallas_fwrite(&max, sizeof(max), 1, vectorFile);
    _pallas_fwrite(&mean, sizeof(mean), 1, vectorFile);
    offset = ftell(valueFile);
    _pallas_fwrite(&offset, sizeof(offset), 1, vectorFile);

    // And write the timestamps to the valueFile
    auto* buffer = new uint64_t[size];
    uint cur_index = 0;
    SubVector* sub_vec = first;
    while (sub_vec) {
      sub_vec->copyToArray(&buffer[sub_vec->starting_index]);
      cur_index += sub_vec->size;
      sub_vec = sub_vec->next;
    }
    pallas_assert(cur_index == size);
    _pallas_compress_write(buffer, size, valueFile);
    delete[] buffer;
  }
}

pallas::LinkedVector::LinkedVector(FILE* vectorFile, FILE* valueFile, const char* valueFilePath) {
  file = valueFile;
  filePath = valueFilePath;
  _pallas_fread(&size, sizeof(size), 1, vectorFile);
  if (size == 0) {
    min, max, mean = 0;
  }
  if (size == 1) {
    _pallas_fread(&min, sizeof(min), 1, vectorFile);
    max, mean = min;
  }
  if (size >= 2) {
    _pallas_fread(&min, sizeof(min), 1, vectorFile);
    _pallas_fread(&max, sizeof(max), 1, vectorFile);
    _pallas_fread(&mean, sizeof(mean), 1, vectorFile);
    _pallas_fread(&offset, sizeof(offset), 1, vectorFile);
  }
  first = nullptr;
  last = nullptr;
}

void pallas::LinkedVector::load_timestamps() {
  if (size < 3) {
    pallas_log(DebugLevel::Debug, "Skipping timestamps from %s\n", filePath);
    return;
  }
  pallas_log(DebugLevel::Debug, "Loading timestamps from %s\n", filePath);

  int ret = fseek(file, offset, 0);
  if (ret == EBADF) {
    file = pallasFileOpen(filePath, "r");
    fseek(file, offset, 0);
  }
  auto temp = _pallas_compress_read(size, file);
  last = new SubVector(size, temp);
  first = last;
}

/**************** Storage Functions ****************/

void pallas_storage_init(pallas::Archive* archive) {
  pallasMkdir(archive->dir_name, 0777);
  pallas_storage_option_init();
}

static const char* base_dirname(pallas::Archive* a) {
  return a->dir_name;
}

static const char* getThreadPath(pallas::Thread* th) {
  char* folderPath = new char[1024];
//  snprintf(folderPath, 1024, "archive_%u/thread_%u", th->archive->id, th->id);
  snprintf(folderPath, 1024, "thread_%u", th->id);
  return folderPath;
}

static const char* pallasGetEventFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/event.pallas", base_dirname, threadPath);
  return filename;
}

static const char* pallasGetEventDurationFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/event_durations.dat", base_dirname, threadPath);
  return filename;
}

static void _pallas_store_attribute_values(pallas::EventSummary* e, FILE* file) {
  _pallas_fwrite(&e->attribute_pos, sizeof(e->attribute_pos), 1, file);
  if (e->attribute_pos > 0) {
    pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", e->attribute_pos);
    if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
      size_t compressedSize = ZSTD_compressBound(e->attribute_pos);
      byte* compressedArray = new byte[compressedSize];
      compressedSize = _pallas_zstd_compress(e->attribute_buffer, e->attribute_pos, compressedArray, compressedSize);
      _pallas_fwrite(&compressedSize, sizeof(compressedSize), 1, file);
      _pallas_fwrite(compressedArray, compressedSize, 1, file);
      delete[] compressedArray;
    } else {
      _pallas_fwrite(e->attribute_buffer, e->attribute_pos, 1, file);
    }
  }
}

static void _pallas_read_attribute_values(pallas::EventSummary* e, FILE* file) {
  _pallas_fread(&e->attribute_pos, sizeof(e->attribute_pos), 1, file);
  e->attribute_buffer_size = e->attribute_pos;
  e->attribute_pos = 0;
  e->attribute_buffer = nullptr;

  if (e->attribute_buffer_size > 0) {
    e->attribute_buffer = new uint8_t[e->attribute_buffer_size];
    if (e->attribute_buffer == nullptr) {
      pallas_error("Cannot allocate memory\n");
    }
    if (pallas::parameterHandler->getCompressionAlgorithm() != pallas::CompressionAlgorithm::None) {
      size_t compressedSize;
      _pallas_fread(&compressedSize, sizeof(compressedSize), 1, file);
      byte* compressedArray = new byte[compressedSize];
      _pallas_fread(e->attribute_buffer, compressedSize, 1, file);
      e->attribute_buffer =
        reinterpret_cast<uint8_t*>(_pallas_zstd_read(e->attribute_buffer_size, compressedArray, compressedSize));
      delete[] compressedArray;
    } else {
      _pallas_fread(e->attribute_buffer, e->attribute_buffer_size, 1, file);
    }
  }
}

static void pallasStoreEvent(pallas::EventSummary& event, FILE* eventFile, FILE* durationFile) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore event %d {.nb_events=%zu}\n", event.id, event.durations->size);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    event.durations->print();
    std::cout << "\n";
  }
  _pallas_fwrite(&event.event, sizeof(pallas::Event), 1, eventFile);
  _pallas_fwrite(&event.attribute_pos, sizeof(event.attribute_pos), 1, eventFile);
  if (event.attribute_pos > 0) {
    pallas_log(pallas::DebugLevel::Debug, "\t\tStore %lu attributes\n", event.attribute_pos);
    _pallas_fwrite(event.attribute_buffer, sizeof(byte), event.attribute_pos, eventFile);
  }
  if (STORE_TIMESTAMPS) {
    event.durations->writeToFile(eventFile, durationFile);
  }
}

static void pallasReadEvent(pallas::EventSummary& event,
                            FILE* eventFile,
                            FILE* durationFile,
                            const char* durationFileName) {
  _pallas_fread(&event.event, sizeof(pallas::Event), 1, eventFile);
  _pallas_fread(&event.attribute_buffer_size, sizeof(event.attribute_buffer_size), 1, eventFile);
  event.attribute_pos = 0;
  event.attribute_buffer = nullptr;
  if (event.attribute_buffer_size > 0) {
    event.attribute_buffer = new byte[event.attribute_buffer_size];
    _pallas_fread(event.attribute_buffer, sizeof(byte), event.attribute_buffer_size, eventFile);
  }
  event.durations = new pallas::LinkedVector(eventFile, durationFile, durationFileName);
}

static const char* pallasGetSequenceFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/sequence.pallas", base_dirname, threadPath);
  return filename;
}

static const char* pallasGetSequenceDurationFilename(const char* base_dirname, pallas::Thread* th) {
  char* filename = new char[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/sequence_durations.dat", base_dirname, threadPath);
  return filename;
}

static void pallasStoreSequence(pallas::Sequence& sequence, FILE* sequenceFile, FILE* durationFile) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id, sequence.size(),
             sequence.durations->size);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    //    th->printSequence(sequence);
    sequence.durations->print();
    std::cout << "\n";
  }
  size_t size = sequence.size();
  _pallas_fwrite(&size, sizeof(size), 1, sequenceFile);
  _pallas_fwrite(sequence.tokens.data(), sizeof(sequence.tokens[0]), sequence.size(), sequenceFile);
  if (STORE_TIMESTAMPS) {
    sequence.durations->writeToFile(sequenceFile, durationFile);
  }
}

static void pallasReadSequence(pallas::Sequence& sequence,
                               FILE* sequenceFile,
                               FILE* durationFile,
                               const char* durationFileName) {
  size_t size;
  _pallas_fread(&size, sizeof(size), 1, sequenceFile);
  sequence.tokens.resize(size);
  _pallas_fread(sequence.tokens.data(), sizeof(pallas::Token), size, sequenceFile);
  if (STORE_TIMESTAMPS) {
    delete sequence.durations;  // durations is created when making a new Sequence
    sequence.durations = new pallas::LinkedVector(sequenceFile, durationFile, durationFileName);
  }
  pallas_log(pallas::DebugLevel::Debug, "\tLoaded sequence %d {.size=%zu, .nb_ts=%zu}\n", sequence.id, sequence.size(),
             sequence.durations->size);
}

static FILE* pallasGetLoopFile(const char* base_dirname, pallas::Thread* th, const char* mode) {
  char filename[1024];
  const char* threadPath = getThreadPath(th);
  snprintf(filename, 1024, "%s/%s/loop.pallas", base_dirname, threadPath);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreLoop(pallas::Loop& loop, FILE* loopFile) {
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug,
               "\tStore loops %d {.nb_loops=%zu, .repeated_token=%d.%d, .nb_iterations:", loop.self_id.id,
               loop.nb_iterations.size(), loop.repeated_token.type, loop.repeated_token.id);
    std::cout << "[";
    for (const auto& i : loop.nb_iterations) {
      std::cout << i << ((&i != &loop.nb_iterations.back()) ? ", " : "]");
    }
    std::cout << "}" << std::endl;
  }
  _pallas_fwrite(&loop.repeated_token, sizeof(loop.repeated_token), 1, loopFile);
  size_t size = loop.nb_iterations.size();
  _pallas_fwrite(&size, sizeof(size), 1, loopFile);
  _pallas_fwrite(loop.nb_iterations.data(), sizeof(uint), loop.nb_iterations.size(), loopFile);
}

static void pallasReadLoop(pallas::Loop& loop, FILE* loopFile) {
  _pallas_fread(&loop.repeated_token, sizeof(loop.repeated_token), 1, loopFile);
  size_t size;
  _pallas_fread(&size, sizeof(size), 1, loopFile);
  loop.nb_iterations.resize(size);
  _pallas_fread(loop.nb_iterations.data(), sizeof(uint), size, loopFile);
  if (pallas::debugLevel >= pallas::DebugLevel::Debug) {
    pallas_log(pallas::DebugLevel::Debug,
               "\tLoad loops %d {.nb_loops=%zu, .repeated_token=%d.%d, .nb_iterations: ", loop.self_id.id,
               loop.nb_iterations.size(), loop.repeated_token.type, loop.repeated_token.id);
    std::cout << "[";
    for (const auto& i : loop.nb_iterations) {
      std::cout << i << ((&i != &loop.nb_iterations.back()) ? ", " : "]");
    }
    std::cout << "}" << std::endl;
  }
}

static FILE* pallasGetStringFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/string.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/string.dat", base_dirname(a), a->id);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreStringGeneric(FILE* file, pallas::String* s) {
  pallas_log(pallas::DebugLevel::Debug, "\tStore String {.ref=%d, .length=%d, .str='%s'}\n",
             s->string_ref, s->length, s->str);

  _pallas_fwrite(&s->string_ref, sizeof(s->string_ref), 1, file);
  _pallas_fwrite(&s->length, sizeof(s->length), 1, file);
  _pallas_fwrite(s->str, sizeof(char), s->length, file);
}

static void pallasStoreString(pallas::Archive* a) {
  if (a->definitions.strings.empty())
    return;

  FILE* file = pallasGetStringFile(a, "w");
  for (auto& s: a->definitions.strings) {
    pallasStoreStringGeneric(file, &s);
  }
  fclose(file);
}

static void pallasReadStringGeneric(FILE* file, pallas::String* s) {
  _pallas_fread(&s->string_ref, sizeof(s->string_ref), 1, file);
  _pallas_fread(&s->length, sizeof(s->length), 1, file);
  s->str = new char[s->length];
  pallas_assert(s->str);
  _pallas_fread(s->str, sizeof(char), s->length, file);
  pallas_log(pallas::DebugLevel::Debug, "\tLoad String {.ref=%d, .length=%d, .str='%s'}\n",
             s->string_ref, s->length, s->str);
}

static void pallasReadString(pallas::Archive* a) {
  if (a->definitions.strings.empty())
    return;

  FILE* file = pallasGetStringFile(a, "r");
  for (auto& s: a->definitions.strings) {
    pallasReadStringGeneric(file, &s);
  }
  fclose(file);
}

static FILE* pallasGetRegionsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/regions.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/regions.dat", base_dirname(a), a->id);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreRegionsGeneric(FILE* file, pallas::Definition* d) {
  if (d->regions.empty())
    return;

  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Regions\n", d->regions.size());
  _pallas_fwrite(d->regions.data(), sizeof(pallas::Region), d->regions.size(), file);
}

static void pallasStoreRegions(pallas::Archive* a) {
  if (a->definitions.regions.empty())
    return;

  FILE* file = pallasGetRegionsFile(a, "w");
  pallasStoreRegionsGeneric(file, &a->definitions);
  fclose(file);
}

static void pallasReadRegionsGeneric(FILE* file, pallas::Definition* d) {
  if (d->regions.empty())
    return;

  _pallas_fread(d->regions.data(), sizeof(pallas::Region), d->regions.size(), file);

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu regions\n", d->regions.size());
}

static void pallasReadRegions(pallas::Archive* a) {
  if (a->definitions.regions.empty())
    return;

  FILE* file = pallasGetRegionsFile(a, "r");
  pallasReadRegionsGeneric(file, &a->definitions);
  fclose(file);
}

static FILE* pallasGetAttributesFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/attributes.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/attributes.dat", base_dirname(a), a->id);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreAttributesGeneric(FILE* file, pallas::Definition* d) {
  if (d->attributes.empty())
    return;

  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu Attributes\n", d->attributes.size());
  for (int i = 0; i < d->attributes.size(); i++) {
    pallas_log(pallas::DebugLevel::Debug, "\t\t[%d] {ref=%d, name=%d, type=%d}\n", i, d->attributes[i].attribute_ref,
               d->attributes[i].name, d->attributes[i].type);
  }

  _pallas_fwrite(d->attributes.data(), sizeof(pallas::Attribute), d->attributes.size(), file);
}

static void pallasStoreAttributes(pallas::Archive* a) {
  if (a->definitions.attributes.empty())
    return;

  FILE* file = pallasGetAttributesFile(a, "w");
  pallasStoreAttributesGeneric(file, &a->definitions);
  fclose(file);
}

static void pallasReadAttributesGeneric(FILE* file, pallas::Definition* d) {
  if (d->attributes.empty())
    return;
  _pallas_fread(d->attributes.data(), sizeof(pallas::Attribute), d->attributes.size(), file);

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu attributes\n", d->attributes.size());
}

static void pallasReadAttributes(pallas::Archive* a) {
  if (a->definitions.attributes.empty())
    return;

  FILE* file = pallasGetAttributesFile(a, "r");
  pallasReadAttributesGeneric(file, &a->definitions);
  fclose(file);
}

static FILE* pallasGetLocationGroupsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/location_groups.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/location_groups.dat", base_dirname(a), a->id);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreLocationGroups(pallas::Archive* a) {
  if (a->location_groups.empty())
    return;

  FILE* file = pallasGetLocationGroupsFile(a, "w");
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu location groupds\n", a->location_groups.size());

  _pallas_fwrite(a->location_groups.data(), sizeof(pallas::LocationGroup), a->location_groups.size(), file);
  fclose(file);
}

static void pallasReadLocationGroups(pallas::Archive* a) {
  if (a->location_groups.empty())
    return;

  FILE* file = pallasGetLocationGroupsFile(a, "r");

  _pallas_fread(a->location_groups.data(), sizeof(pallas::LocationGroup), a->location_groups.size(), file);
  fclose(file);

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %zu location_groups\n", a->location_groups.size());
}

static FILE* pallasGetLocationsFile(pallas::Archive* a, const char* mode) {
  char filename[1024];
  if (a->id == PALLAS_MAIN_LOCATION_GROUP_ID)
    snprintf(filename, 1024, "%s/locations.dat", base_dirname(a));
  else
    snprintf(filename, 1024, "%s/archive_%u/locations.dat", base_dirname(a), a->id);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreLocations(pallas::Archive* a) {
  if (a->locations.empty())
    return;

  FILE* file = pallasGetLocationsFile(a, "w");
  pallas_log(pallas::DebugLevel::Debug, "\tStore %zu locations\n", a->locations.size());

  for (auto& l : a->locations) {
    pallas_assert(l.id != PALLAS_THREAD_ID_INVALID);
  }

  _pallas_fwrite(a->locations.data(), sizeof(pallas::Location), a->locations.size(), file);
  fclose(file);
}

static void pallasReadLocations(pallas::Archive* a) {
  if (a->locations.empty())
    return;

  FILE* file = pallasGetLocationsFile(a, "r");

  _pallas_fread(a->locations.data(), sizeof(pallas::Location), a->locations.size(), file);
  fclose(file);

  pallas_log(pallas::DebugLevel::Debug, "\tLoad %lu locations\n", a->locations.size());
}

static FILE* pallasGetThreadFile(const char* dir_name, pallas::Thread* thread, const char* mode) {
  char filename[1024];
  const char* threadPath = getThreadPath(thread);
  snprintf(filename, 1024, "%s/%s/thread.pallas", dir_name, threadPath);
  return pallasFileOpen(filename, mode);
}

static void pallasStoreThread(const char* dir_name, pallas::Thread* th) {
  if (th->nb_events == 0) {
    pallas_log(pallas::DebugLevel::Verbose, "\tSkipping Thread %u {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n",
               th->id, th->nb_events, th->nb_sequences, th->nb_loops);
    abort();
  }

  FILE* token_file = pallasGetThreadFile(dir_name, th, "w");

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);

  _pallas_fwrite(&th->id, sizeof(th->id), 1, token_file);
  _pallas_fwrite(&th->archive->id, sizeof(th->archive->id), 1, token_file);

  _pallas_fwrite(&th->nb_events, sizeof(th->nb_events), 1, token_file);
  _pallas_fwrite(&th->nb_sequences, sizeof(th->nb_sequences), 1, token_file);
  _pallas_fwrite(&th->nb_loops, sizeof(th->nb_loops), 1, token_file);

  fclose(token_file);

  const char* eventFilename = pallasGetEventFilename(dir_name, th);
  const char* eventDurationFilename = pallasGetEventDurationFilename(dir_name, th);
  FILE* eventFile = pallasFileOpen(eventFilename, "w");
  FILE* eventDurationFile = pallasFileOpen(eventDurationFilename, "w");
  for (int i = 0; i < th->nb_events; i++) {
    pallasStoreEvent(th->events[i], eventFile, eventDurationFile);
  }
  fclose(eventFile);
  fclose(eventDurationFile);

  const char* sequenceFilename = pallasGetSequenceFilename(dir_name, th);
  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(dir_name, th);
  FILE* sequenceFile = pallasFileOpen(sequenceFilename, "w");
  FILE* sequenceDurationFile = pallasFileOpen(sequenceDurationFilename, "w");
  for (int i = 0; i < th->nb_sequences; i++) {
    pallasStoreSequence(*th->sequences[i], sequenceFile, sequenceDurationFile);
  }
  fclose(sequenceFile);
  fclose(sequenceDurationFile);

  FILE* loopFile = pallasGetLoopFile(dir_name, th, "w");
  for (int i = 0; i < th->nb_loops; i++)
    pallasStoreLoop(th->loops[i], loopFile);
  fclose(loopFile);
  pallas_log(pallas::DebugLevel::Debug, "Average compression ratio: %.2f\n",
             (numberRawBytes + .0) / numberCompressedBytes);
}

void pallas::Thread::finalizeThread() {
  pallasStoreThread(archive->dir_name, this);
}

static void pallasReadThread(pallas::Archive* global_archive, pallas::Thread* th, pallas::ThreadId thread_id) {
  th->id = thread_id;
  FILE* threadFile = pallasGetThreadFile(global_archive->dir_name, th, "r");
  _pallas_fread(&th->id, sizeof(th->id), 1, threadFile);
  pallas::LocationGroupId archive_id;
  _pallas_fread(&archive_id, sizeof(archive_id), 1, threadFile);
  th->archive = pallasGetArchive(global_archive, archive_id);

  _pallas_fread(&th->nb_events, sizeof(th->nb_events), 1, threadFile);
  th->nb_allocated_events = th->nb_events;
  th->events = new pallas::EventSummary[th->nb_allocated_events];

  _pallas_fread(&th->nb_sequences, sizeof(th->nb_sequences), 1, threadFile);
  th->nb_allocated_sequences = th->nb_sequences;
  th->sequences = new pallas::Sequence*[th->nb_allocated_sequences];
  for (int i = 0; i < th->nb_sequences; i++) {
    th->sequences[i] = new pallas::Sequence;
  }

  _pallas_fread(&th->nb_loops, sizeof(th->nb_loops), 1, threadFile);
  th->nb_allocated_loops = th->nb_loops;
  th->loops = new pallas::Loop[th->nb_allocated_loops];

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d events\n", th->nb_events);
  const char* eventFilename = pallasGetEventFilename(global_archive->dir_name, th);
  const char* eventDurationFilename = pallasGetEventDurationFilename(global_archive->dir_name, th);
  FILE* eventFile = pallasFileOpen(eventFilename, "r");
  FILE* eventDurationFile = pallasFileOpen(eventDurationFilename, "r");
  for (int i = 0; i < th->nb_events; i++) {
    th->events[i].id = i;
    pallasReadEvent(th->events[i], eventFile, eventDurationFile, eventDurationFilename);
  }
  fclose(eventFile);

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d sequences\n", th->nb_sequences);
  const char* sequenceFilename = pallasGetSequenceFilename(global_archive->dir_name, th);
  const char* sequenceDurationFilename = pallasGetSequenceDurationFilename(global_archive->dir_name, th);
  FILE* sequenceFile = pallasFileOpen(sequenceFilename, "r");
  FILE* sequenceDurationFile = pallasFileOpen(sequenceDurationFilename, "r");
  for (int i = 0; i < th->nb_sequences; i++) {
    th->sequences[i]->id = i;
    pallasReadSequence(*th->sequences[i], sequenceFile, sequenceDurationFile, sequenceDurationFilename);
  }
  fclose(sequenceFile);

  pallas_log(pallas::DebugLevel::Verbose, "Reading %d loops\n", th->nb_loops);
  FILE* loopFile = pallasGetLoopFile(global_archive->dir_name, th, "r");
  for (int i = 0; i < th->nb_loops; i++) {
    th->loops[i].self_id = PALLAS_LOOP_ID(i);
    pallasReadLoop(th->loops[i], loopFile);
  }
  fclose(loopFile);
  fclose(threadFile);

  pallas_log(pallas::DebugLevel::Verbose, "\tThread %u: {.nb_events=%d, .nb_sequences=%d, .nb_loops=%d}\n", th->id,
             th->nb_events, th->nb_sequences, th->nb_loops);
}

void pallas_storage_finalize_thread(pallas::Thread* thread) {
  if (!thread)
    return;
  pallasStoreThread(thread->archive->dir_name, thread);
}

void pallas_storage_finalize(pallas::Archive* archive) {
  if (!archive)
    return;

  char* fullpath;
  size_t fullpath_len;
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    fullpath_len = strlen(archive->dir_name) + strlen(archive->trace_name) + strlen("%s/%s.pallas");
    fullpath = new char[fullpath_len];
    snprintf(fullpath, fullpath_len, "%s/%s.pallas", archive->dir_name, archive->trace_name);
  } else {
    fullpath_len = strlen(archive->dir_name) + 32 + strlen("%s/archive_%u/archive.pallas");
    fullpath = new char[fullpath_len];
    snprintf(fullpath, fullpath_len, "%s/archive_%u/archive.pallas", archive->dir_name, archive->id);
  }

  FILE* f = pallasFileOpen(fullpath, "w");
  delete[] fullpath;
  _pallas_fwrite(&archive->id, sizeof(pallas::LocationGroupId), 1, f);
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    uint8_t version = PALLAS_ABI_VERSION;
    _pallas_fwrite(&version, sizeof(version), 1, f);
    pallas::parameterHandler->writeToFile(f);
  }
  size_t size = archive->definitions.strings.size();
  _pallas_fwrite(&size, sizeof(size), 1, f);
  size = archive->definitions.regions.size();
  _pallas_fwrite(&size, sizeof(size), 1, f);
  size = archive->definitions.attributes.size();
  _pallas_fwrite(&size, sizeof(size), 1, f);
  size = archive->location_groups.size();
  _pallas_fwrite(&size, sizeof(size), 1, f);
  size = archive->locations.size();
  _pallas_fwrite(&size, sizeof(size), 1, f);
  _pallas_fwrite(&archive->nb_threads, sizeof(int), 1, f);
  //  _pallas_fwrite(&COMPRESSION_OPTIONS, sizeof(COMPRESSION_OPTIONS), 1, f);
  _pallas_fwrite(&STORE_HASHING, sizeof(STORE_HASHING), 1, f);
  _pallas_fwrite(&STORE_TIMESTAMPS, sizeof(STORE_TIMESTAMPS), 1, f);

  pallasStoreString(archive);
  pallasStoreRegions(archive);
  pallasStoreAttributes(archive);

  pallasStoreLocationGroups(archive);
  pallasStoreLocations(archive);

  fclose(f);
}

static char* _archive_filename(pallas::Archive* global_archive, pallas::LocationGroupId id) {
  if (id == PALLAS_MAIN_LOCATION_GROUP_ID)
    return strdup(global_archive->trace_name);

  int tracename_len = strlen(global_archive->trace_name) + 1;
  int extension_index = tracename_len - 8;
  pallas_assert(strcmp(&global_archive->trace_name[extension_index], ".pallas") == 0);

  char trace_basename[tracename_len];
  strncpy(trace_basename, global_archive->trace_name, extension_index);
  trace_basename[extension_index] = '\0';

  int len = strlen("archive") * 2 + 32 + 5;
  char* result = new char[len];
  snprintf(result, len, "archive_%d/archive.pallas", id);
  return result;
}

char* pallas_archive_fullpath(char* dir_name, char* trace_name) {
  int len = strlen(dir_name) + strlen(trace_name) + 2;
  char* fullpath = new char[len];
  snprintf(fullpath, len, "%s/%s", dir_name, trace_name);
  return fullpath;
}

void pallas::ParameterHandler::writeToFile(FILE* file) const {
  _pallas_fwrite(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fwrite(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fwrite(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fwrite(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fwrite(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fwrite(&timestampStorage, sizeof(timestampStorage), 1, file);
}

void pallas::ParameterHandler::readFromFile(FILE* file) {
  _pallas_fread(&compressionAlgorithm, sizeof(compressionAlgorithm), 1, file);
  _pallas_fread(&encodingAlgorithm, sizeof(encodingAlgorithm), 1, file);
  _pallas_fread(&zstdCompressionLevel, sizeof(zstdCompressionLevel), 1, file);
  _pallas_fread(&loopFindingAlgorithm, sizeof(loopFindingAlgorithm), 1, file);
  _pallas_fread(&maxLoopLength, sizeof(maxLoopLength), 1, file);
  _pallas_fread(&timestampStorage, sizeof(timestampStorage), 1, file);
}

static void pallasReadArchive(pallas::Archive* global_archive,
                              pallas::Archive* archive,
                              char* dir_name,
                              char* trace_name) {
  archive->fullpath = pallas_archive_fullpath(dir_name, trace_name);
  archive->dir_name = dir_name;
  archive->trace_name = trace_name;
  archive->global_archive = global_archive;
  archive->nb_archives = 0;
  archive->nb_allocated_archives = 1;
  archive->archive_list = new pallas::Archive*();
  archive->definitions = pallas::Definition();
  if (archive->archive_list == nullptr) {
    pallas_error("Failed to allocate memory\n");
  }

  pallas_log(pallas::DebugLevel::Debug, "Reading archive {.dir_name='%s', .trace='%s'}\n", archive->dir_name,
             archive->trace_name);

  FILE* f = pallasFileOpen(archive->fullpath, "r");

  _pallas_fread(&archive->id, sizeof(pallas::LocationGroupId), 1, f);
  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    // Version checking
    uint8_t abi_version;
    _pallas_fread(&abi_version, sizeof(abi_version), 1, f);
    pallas_assert_always(abi_version == PALLAS_ABI_VERSION);
    pallas::parameterHandler = new pallas::ParameterHandler();
    pallas::parameterHandler->readFromFile(f);
  }
  size_t size;

  _pallas_fread(&size, sizeof(size), 1, f);
  archive->definitions.strings.resize(size);
  _pallas_fread(&size, sizeof(size), 1, f);
  archive->definitions.regions.resize(size);
  _pallas_fread(&size, sizeof(size), 1, f);
  archive->definitions.attributes.resize(size);
  _pallas_fread(&size, sizeof(size), 1, f);
  archive->location_groups.resize(size);
  _pallas_fread(&size, sizeof(size), 1, f);
  archive->locations.resize(size);

  _pallas_fread(&archive->nb_threads, sizeof(int), 1, f);

  archive->threads = (pallas::Thread**)calloc(sizeof(pallas::Thread*), archive->nb_threads);
  archive->nb_allocated_threads = archive->nb_threads;

  //  _pallas_fread(&COMPRESSION_OPTIONS, sizeof(COMPRESSION_OPTIONS), 1, f);
  _pallas_fread(&STORE_HASHING, sizeof(STORE_HASHING), 1, f);
  _pallas_fread(&STORE_TIMESTAMPS, sizeof(STORE_TIMESTAMPS), 1, f);

  char* store_timestamps_str = getenv("STORE_TIMESTAMPS");
  if (store_timestamps_str && strcmp(store_timestamps_str, "FALSE") == 0) {
    STORE_TIMESTAMPS = 0;
  }
  archive->store_timestamps = STORE_TIMESTAMPS;

  pallasReadString(archive);

  pallasReadRegions(archive);
  pallasReadAttributes(archive);

  if (!archive->location_groups.empty()) {
    pallasReadLocationGroups(archive);
  }

  if (!archive->locations.empty()) {
    pallasReadLocations(archive);
  }

  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    global_archive = archive;
  }
  for (auto& location : archive->locations) {
    pallas_assert(location.id != PALLAS_THREAD_ID_INVALID);
    pallasLoadThread(global_archive, location.id);
  }


  fclose(f);
}

static pallas::Archive* pallasGetArchive(pallas::Archive* global_archive,
                                         pallas::LocationGroupId archive_id,
                                         bool print_warning) {
  /* check if archive_id is already known */
  for (int i = 0; i < global_archive->nb_archives; i++) {
    if (global_archive->archive_list[i]->id == archive_id) {
      return global_archive->archive_list[i];
    }
  }

  /* not found. we need to read the archive */
  auto* arch = new pallas::Archive();
  char* filename = _archive_filename(global_archive, archive_id);
  char* fullpath = pallas_archive_fullpath(global_archive->dir_name, filename);
  if (access(fullpath, R_OK) < 0) {
    if (print_warning)
      pallas_warn("I can't read %s: %s\n", fullpath, strerror(errno));
    free(fullpath);
    return nullptr;
  }

  pallas_log(pallas::DebugLevel::Verbose, "Reading archive %s\n", fullpath);
  delete[] fullpath;

  while (global_archive->nb_archives >= global_archive->nb_allocated_archives) {
    INCREMENT_MEMORY_SPACE(global_archive->archive_list, global_archive->nb_allocated_archives, pallas::Archive*);
  }

  pallasReadArchive(global_archive, arch, strdup(global_archive->dir_name), filename);

  int index = global_archive->nb_archives++;
  global_archive->archive_list[index] = arch;

  return arch;
}

void pallasLoadThread(pallas::Archive* archive, pallas::ThreadId thread_id) {
  for (int i = 0; i < archive->nb_threads; i++) {
    if (archive->threads[i]->id == thread_id) {
      /* thread_id is already loaded */
      return;
    }
  }

  while (archive->nb_threads >= archive->nb_allocated_threads) {
    INCREMENT_MEMORY_SPACE(archive->threads, archive->nb_allocated_threads, pallas::Thread*);
  }

  int index = archive->nb_threads++;
  archive->threads[index] = new pallas::Thread();
  pallasReadThread(archive, archive->threads[index], thread_id);
  pallas_assert(archive->threads[index]->nb_events > 0);
}

void pallas_read_main_archive(pallas::Archive* archive, char* main_filename) {
  auto* temp_main_filename = strdup(main_filename);
  char* trace_name = strdup(basename(temp_main_filename));
  char* dir_name = strdup(dirname(temp_main_filename));
  free(temp_main_filename);

  pallasReadArchive(nullptr, archive, dir_name, trace_name);

  pallas::Archive* global_archive = archive->global_archive;

  if (archive->id == PALLAS_MAIN_LOCATION_GROUP_ID) {
    global_archive = archive;
  }

  for (auto& location : archive->locations) {
    pallasLoadThread(global_archive, location.id);
  }

  for (auto& location_group : archive->location_groups) {
    pallasGetArchive(global_archive, location_group.id, false);
  }
  for (auto& location : archive->locations) {
    pallasGetArchive(global_archive, location.id, false);
  }
}

/* -*-
   mode: c;
   c-file-style: "k&r";
   c-basic-offset 2;
   tab-width 2 ;
   indent-tabs-mode nil
   -*- */
