# Pallas

Pallas provides an interface to write and read trace data.

## Building
You need to have ZSTD installed before you try to build this.
To build and install, simply run:
```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALLATION_DIR
make && make install
```

This will build and install `pallas_info` and `pallas_print`,
which are tools made to read Pallas traces, as well as the Pallas library, and a modified OTF2 library.

If you want to enable SZ and ZFP, you should install them, and then add `-DSZ_ROOT_DIR=<your SZ installation>`
and `-DZFP_ROOT_DIR=<your ZFP installation>` to the cmake command line. Documentation is built automatically if Doxygen is installed.

## Usage
### In your application
These few lines are all you need
```C
// In C
#include <pallas/pallas.h>
#include <pallas/pallas_write.h>

int main() {
    // Setup everything
    GlobalArchive* global_archive = pallas_global_archive_new(); // Create the main trace
    pallas_write_global_archive_open(global_archive, "<your trace name>", "main");
    // The Global Archive is where all the Strings, 
    // information about Threads and Processes, and Regions are stored.
    
    pallas_archive_register_string(...)     // Register a String
    pallas_write_define_location_group(<processID>) // Register a LocationGroup
    Archive* archive = pallas_new_archive();
    pallas_write_archive_open(global_archive, archive, <processID>);
    // That creates an Archive, which is where you'll store local events.
            
    pallas_write_define_location(<threadID>)       // Register a Location
    ThreadWriter thread_writer;
    pallas_write_thread_open(global_archive, &thread_writer, <threadID>);
    // A ThreadWriter is the interface made to log some events
    
    // Start logging
    pallas_record_generic(&thread_writer, <custom Attribute>, <timestamp>, <name>);
    
    // Write the trace to file
    pallas_write_thread_close(thread_writer);
    pallas_write_global_archive_close(global_archive);
}
```
```CPP
// In C++
#include <pallas/pallas.h>
#include <pallas/pallas_write.h>
namespace pallas;
int main() {
    // Setup everything
    GlobalArchive globalArchive = Archive(); // Create the main trace
    globalArchive.openGlobal("<your trace name>", "main");
    // The Global Archive is where all the Strings, 
    // information about Threads and Processes, and Regions are stored.
    
    globalArchive.addString(...)                   // Register a String
    globalArchive.addLocationGroup(<processID>)    // Register a LocationGroup
    Archive archive = Archive();
    archive.open(globalArchive, <processID>);
    // That creates an Archive, which is where you'll store local events.

    globalArchive.addLocation(<threadID>)         // Register a Location
    ThreadWriter threadWriter;
    threadWriter.openThread(globalArchive, <threadID>);
    // A ThreadWriter is the interface made to log some events

    // Start logging
    pallas_record_generic(&threadWriter, <custom Attribute>, <timestamp>, <name>);
    
    // Write the trace to file
    threadWriter.close();
    globalArchive.close();
}
```




### Using EZTrace

After compiling Pallas and its OTF2 API, install [EZTrace](https://eztrace.gitlab.io/eztrace).
Make sure to build it from source, and to use the Pallas OTF2 library, not the normal OTF2 library.
You can check `which otf2-config` to see if you have the correct one. If not, check your PATH and LD_LIBRARY_PATH variables.

Make sure to enable the relevant ezTrace modules.
Then trace any program by running `mpirun -np N eztrace -t <your modules> <your programm>`.
The trace file will be generated in the `<your programm>_trace` folder.
You can then read it using `pallas_print <your programm>_trace/eztrace_log.pallas`

## About

Pallas implements a subset of the [OTF2](https://www.vi-hps.org/projects/score-p) API.
It also implements the [Murmur3 hashing function](https://github.com/PeterScott/murmur3).

## Configuration

A config file can be given to Pallas with the PALLAS_CONFIG_PATH environment variable.
If that variable is empty, Pallas will try to load a pallas.config in the current directory.
If that file does not exist, a default config will be loaded.
That file is a mirror image of the ParameterHandler class in `libraries/pallas/src/ParameterHandler.h`
An example config file is given here as pallas.config, each line has one `key=value` pair.

Here are the configuration options with specific values:

- `compressionAlgorithm`: Specifies which compression algorithm is used for storing the timestamps. Its values are:
  - `None`
  - `ZSTD`
  - `SZ`
  - `ZFP`
  - `Histogram`
  - `ZSTD_Histogram`
- `encodingAlgorithm`: Specifies which encoding algorithm is used for storing the timestamps. If the specified
  compression algorithm is lossy, this is defaulted to None. Its values are:
  - `None`
  - `Masking`
  - `LeadingZeroes`
- `loopFindingAlgorithm`: Specifies what loop-finding algorithm is used. Its values are:
  - `None`
  - `Basic`
  - `BasicTruncated`
  - `Filter`

Here are the configuration options with number values:

- `zstdCompressionLevel`: Specifies the compression level used by ZSTD. Integer.
- `maxLoopLength`: Specifies the maximum loop length, if using a truncated loop finding algorithm. Integer.

You can also override each of these configuration manually with an environment variable.
Here are the default values for each of them:

| Config Key Name      | Env Variable Name   | Default Value  |
|----------------------|---------------------|----------------|
| compressionAlgorithm | PALLAS_COMPRESSION  | None           |
| encodingAlgorithm    | PALLAS_ENCODING     | None           |
| loopFindingAlgorithm | PALLAS_LOOP_FINDING | BasicTruncated |
| zstdCompressionLevel | PALLAS_ZSTD_LVL     | 3              |
| maxLoopLength        | PALLAS_LOOP_LENGTH  | 100            |

## Contributing

Contribution to Pallas are welcome. Just send us a pull request.

## License
Pallas is distributed under the terms of both the BSD 3-Clause license.

See [LICENSE](LICENSE) for details
