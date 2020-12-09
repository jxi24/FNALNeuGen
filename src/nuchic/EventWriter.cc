#include "nuchic/EventWriter.hh"
#include "nuchic/Event.hh"
#include "nuchic/Particle.hh"
#include "fmt/format.h"

nuchic::NuchicWriter::NuchicWriter(const std::string &filename) : toFile{true} {
    m_out = new std::ofstream(filename);
}

void nuchic::NuchicWriter::WriteHeader(const std::string &filename) {
    *m_out << "Nuchic Version: 1.0.0\n";
    *m_out << fmt::format("{0:-^40}\n\n", "");

    std::ifstream input(filename);
    std::string line;
    while(std::getline(input, line)) {
        *m_out << fmt::format("{}\n", line);
    }
    *m_out << fmt::format("{0:-^40}\n\n", "");
}

void nuchic::NuchicWriter::Write(const Event &event) {
    *m_out << fmt::format("Event: {}\n", ++nEvents);
    *m_out << fmt::format("  Particles:\n");
    for(const auto &part : event.Particles()) {
        *m_out << fmt::format("  - {}\n", part);
    }
    *m_out << fmt::format("  Weight: {}\n\n", event.Weight());
}