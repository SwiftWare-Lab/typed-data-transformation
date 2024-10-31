//
// Created by kazem on 10/31/24.
//

#ifndef BIG_DATA_UTILS_H
#define BIG_DATA_UTILS_H

#include <iostream>
#include <argparse/argparse.hpp>

namespace swiftware::bigdata {
    struct CompressionParameters {
        std::string compression_type, input_file, output_file;
        int compression_level;

        //TODO others
        CompressionParameters() {
            compression_level = 0;
        }
    };

    CompressionParameters *get_args(int argc, char **argv, argparse::ArgumentParser *program) {
        // TODO add yours,
        program->add_argument("-i", "--input")
                .help("Input file")
                .required();
        program->add_argument("-o", "--output")
                .help("Output file");
        program->add_argument("-t", "--type")
                .help("Type of operation");
        program->add_argument("-l", "--level")
                .help("Compression level");
        try {
            program->parse_args(argc, argv);
        } catch (const std::runtime_error &err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            exit(1);
        }
        auto args = new CompressionParameters();
        if (program->is_used("--type"))
            args->compression_type = program->get<std::string>("--type");
        if (program->is_used("--input"))
            args->input_file = program->get<std::string>("--input");
        if (program->is_used("--output"))
            args->output_file = program->get<std::string>("--output");
        if (program->is_used("--level"))
            args->compression_level = program->get<int>("--level");
        return args;
    }
}

#endif //BIG_DATA_UTILS_H
