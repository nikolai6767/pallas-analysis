/*
 * Copyright (C) Telecom SudParis
 * See LICENSE in top-level directory.
 */
/** @file
 * A custom type of Linked-List. This takes into account the fact that we never go remove anything from timestamps
 * vector.
 */
#pragma once

#include "pallas_dbg.h"
#ifndef __cplusplus
#include <stdint.h>
#endif
#ifdef __cplusplus
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
/** Default size for creating Vectors and SubVectors.*/
#define DEFAULT_VECTOR_SIZE 1000
namespace pallas {

/**
 * Classic linked array list. Sub-arrays are implemented as a subclass
 */
class LinkedVector {
   public:
    /** Number of element stored in the vector.  */
    size_t size = 0;
    /**
     * Adds a new element at the end of the vector, after its current last element.
     *
     * @param val Value to be added.
     * @return Reference to the new element.
     */
    uint64_t* add(uint64_t val);
    /**
     * Returns a reference to the element at specified location `pos`, with bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& at(size_t pos);
    /**
     * Returns a reference to the element at specified location `pos`, without bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& operator[](size_t pos);
    /**
     * Returns a reference to the first element in the vector.
     * @return Reference to the first element.
     */
    [[nodiscard]] uint64_t& front();
    /**
     * Returns a reference to the last element in the vector.
     * @return Reference to the last element.
     */
    [[nodiscard]] uint64_t& back();
    /**
     * Loads the timestamps / durations from filePath.
     */
    void load_timestamps();
    /**
     * Frees the data contained in the vector, but keeps the references needed to load them again.
     */
    void free_data();

    /**
     * Returns a representation of the vector as a string, for example: "[10, 10000, 3141]"
     */
    std::string to_string();
    /**
     * Writes the vector to the given files.
     * @param infoFile File where information about the vector is stored.
     * @param dataFile  File where most of the data are stored.
     */
    void write_to_file(FILE* infoFile, FILE* dataFile);

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;
    /** Offset in the file. */
    long offset = 0;
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
       public:
        /** Number of elements stored in the vector. */
        size_t size = 0;

        /** Number of elements this vector has allocated. */
        size_t allocated = DEFAULT_VECTOR_SIZE;

        /** Array of elements. Currently only used on uint64_t */
        uint64_t* array = nullptr;

        /** Next SubArray in the Vector. nullptr if last. */
        SubArray* next = nullptr;

        /** Previous SubArray in the Vector. nullptr if first. */
        SubArray* previous = nullptr;

        /** Starting index of this SubVector. */
        size_t starting_index = 0;

        /**
         * Adds a new element at the end of the vector, after its current last element.
         *
         * @param val Value to be added.
         * @return Reference to the new element.
         */
        uint64_t* add(uint64_t val);

        /**
         * Returns a reference to the element at specified location `pos`, with bounds checking.
         * @param pos Position of the element in the array.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& at(size_t pos) const;

        /**
         * Returns a reference to the element at specified location `pos`, without bounds checking.
         * @param pos Position of the element in the LinkedVector.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& operator[](size_t pos) const;

        /**
         * Copies the values in array to given_array.
         * @param given_array An allocated array of correct size.
         */
        void copy_to_array(uint64_t* given_array) const;
        ~SubArray();

        /**
         * Construct a SubArray of a given size.
         * @param size Size of the SubVector.
         * @param previous Previous SubArray.
         */
        explicit SubArray(size_t size, SubArray* previous = nullptr);

        /**
         * Construct a SubArray from a given already allocated array, and its size.
         * @param size: Size of `array`.
         * @param array Allocated array of values.
         */
        SubArray(size_t size, uint64_t* array);
    };

    /** First array list in the linked array list structure.*/
    SubArray* first;
    /** Last array list in the linked array list structure.*/
    SubArray* last;

   public:
    /**
     * Creates a new LinkedVector.
     */
    LinkedVector();
    /** Creates a new LinkedVector from a file. Doesn't actually load it until and element is accessed. */
    LinkedVector(FILE* vectorFile, const char* valueFilePath);

    /**
     * Classic destructor. Calls free_data().
     */
    ~LinkedVector();
};

class LinkedDurationVector {
   public:
    /** Number of element stored in the vector.  */
    size_t size = 0;

    /**
     * Adds a new element at the end of the vector, after its current last element.
     * Updates mean, min and max.
     *
     * @param val Value to be added.
     * @return Pointer to the new element.
     */
    uint64_t* add(uint64_t val);
    /**
     * Returns a reference to the element at specified location `pos`, with bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& at(size_t pos);
    /**
     * Returns a reference to the element at specified location `pos`, without bounds checking.
     * Loads the vector from the file if needed.
     * @param pos Position of the element in the vector.
     * @return Reference to the requested element.
     */
    [[nodiscard]] uint64_t& operator[](size_t pos);
    /**
     * Returns a reference to the first element in the vector.
     * @return Reference to the first element.
     */
    [[nodiscard]] uint64_t& front();
    /**
     * Returns a reference to the last element in the vector.
     * @return Reference to the last element.
     */
    [[nodiscard]] uint64_t& back();
    /**
     * Loads the timestamps / durations from filePath.
     */
    void load_timestamps();
    /**
     * Frees the data contained in the vector, but keeps the references needed to load them again.
     */
    void free_data();

    /**
     * Returns a representation of the vector as a string, for example: "[10, 10000, 3141] { min, mean, max }"
     */
    std::string to_string();

    /**
     * Writes the vector to the given files.
     * If size >= 4, we do the following:
     *    - To vectorFile, we write [size, min, max, mean, offset] in that order.\n
     *    - To valueFile, we write the array.\n
     * If size <= 3, we don't write anything to valueFile.
     * Instead, we write [size] + array to vectorFile.\n
     * @param vectorFile File where metadata is stored.
     * @param valueFile  File where data is stored (most of the time).
     */
    void write_to_file(FILE* vectorFile, FILE* valueFile);

   private:
    /** Path to the file storing this vector. */
    const char* filePath = nullptr;
    /** Offset in the file. */
    long offset = 0;
    /**
     * A fixed-sized array functioning as a node in a linked array list.
     */
    class SubArray {
       public:
        /** Number of elements stored in the vector. */
        size_t size = 0;

        /** Number of elements this vector has allocated. */
        size_t allocated = DEFAULT_VECTOR_SIZE;

        /** Array of elements. Currently only used on uint64_t */
        uint64_t* array = nullptr;

        /** Next SubArray in the Vector. nullptr if last. */
        SubArray* next = nullptr;

        /** Previous SubArray in the Vector. nullptr if first. */
        SubArray* previous = nullptr;

        /** Starting index of this SubVector. */
        size_t starting_index = 0;

        /**
         * Updates the min/max/mean, taking into account all the items from 0 to size-1.
         *
         * This is because we assume the last element isn't a duration, but a timestamp.
         */
        void update_statistics();

       public:
        /** Max element stored in the array. */
        uint64_t min = UINT64_MAX;

        /** Min element stored in the array. */
        uint64_t max = 0;

        /** Mean of all the elements in the array. */
        uint64_t mean = 0;

        /**
         * Adds a new element at the end of the vector, after its current last element.
         * Updates mean, min and max.
         *
         * @param val Value to be added.
         * @return Pointer to the new element.
         */
        uint64_t* add(uint64_t val);

        /** Does the final calculation for updating the statistics in that vector.*/
        void final_update_statistics();

        /**
         * Returns a reference to the element at specified location `pos`, with bounds checking.
         * @param pos Position of the element in the array.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& at(size_t pos) const;

        /**
         * Returns a reference to the element at specified location `pos`, without bounds checking.
         * @param pos Position of the element in the LinkedVector.
         * @return Reference to the requested element.
         */
        [[nodiscard]] uint64_t& operator[](size_t pos) const;

        /**
         * Copies the values in array to given_array.
         * @param given_array An allocated array of correct size.
         */
        void copy_to_array(uint64_t* given_array) const;
        ~SubArray();

        /**
         * Construct a SubArray of a given size.
         * @param size Size of the SubVector.
         * @param previous Previous SubArray.
         */
        explicit SubArray(size_t size, SubArray* previous = nullptr);

        /**
         * Construct a SubArray from a given already allocated array, and its size.
         * @param size: Size of `array`.
         * @param array Allocated array of values.
         */
        SubArray(size_t size, uint64_t* array);
    };

    /** First array list in the linked array list structure.*/
    SubArray* first;
    /** Last array list in the linked array list structure.*/
    SubArray* last;
    /**
     * Updates the min/max/mean, taking into account all the items from 0 to size-1.
     *
     * This is because we assume the last element isn't a duration, but a timestamp.
     */
    void update_statistics();

   public:
    /** Does the final calculation for updating the statistics in that vector.*/
    void final_update_statistics();
    ~LinkedDurationVector();
    /** Max element stored in the vector. */
    uint64_t min = UINT64_MAX;
    /** Min element stored in the vector. */
    uint64_t max = 0;
    /** Mean of all the elements in the vector. */
    uint64_t mean = 0;
    /**
     * Loads a LinkedDurationVector from a file.
     * Only loads the statistics, doesn't load the timestamps until they're accessed.
     */
    LinkedDurationVector(FILE* vectorFile, const char* valueFilePath);

    /**
     * Creates a new LinkedDurationVector.
     */
    LinkedDurationVector();
};
}  // namespace pallas

#else
typedef struct LinkedVector {
} LinkedVector;

typedef struct LinkedDurationVector {
} LinkedDurationVector;
#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */
