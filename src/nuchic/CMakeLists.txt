add_library(utilities SHARED
    FourVector.cc
    Interpolation.cc
    Particle.cc
    ThreeVector.cc
    Utilities.cc
)
target_include_directories(utilities PRIVATE ${NUCHIC_INCLUDES} PUBLIC ${Python_INCLUDE_DIRS})

set(NUCHIC_LIBS ${NUCHIC_LIBS} utilities)

add_library(physics SHARED
    Cascade.cc
    Interactions.cc
    InteractionsFactory.cc
    Nucleus.cc
)
target_include_directories(physics PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(physics PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

set(NUCHIC_LIBS ${NUCHIC_LIBS} physics)

pybind11_add_module(vectors MODULE
    VectorModule.cc
)
target_include_directories(vectors PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(vectors PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

pybind11_add_module(interpolation MODULE
    InterpolationModule.cc
)
target_include_directories(interpolation PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(interpolation PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

pybind11_add_module(particle MODULE
    ParticleModule.cc
)
target_include_directories(particle PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(particle PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

pybind11_add_module(nucleus MODULE
    NucleusModule.cc
)
target_include_directories(nucleus PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(nucleus PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

pybind11_add_module(interactions MODULE
    InteractionsModule.cc
)
target_include_directories(interactions PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(interactions PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})

pybind11_add_module(cascade MODULE
    CascadeModule.cc
)
target_include_directories(cascade PRIVATE ${NUCHIC_INCLUDES})
target_link_libraries(cascade PRIVATE ${NUCHIC_LIBS} ${HDF5_LIBRARIES})
