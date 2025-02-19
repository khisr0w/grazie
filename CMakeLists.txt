cmake_minimum_required(VERSION 3.16)

project(grazie)
set(EXE_NAME "run")

set(SOURCES
    src/main.c
)

# set(HEADERS
#     glad/glad.h
# )

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(BIN_DIR debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(BIN_DIR release)
endif()

add_executable(${EXE_NAME} ${SOURCES})
# TODO(abid): `GRAZIE_PLT_WIN` must be moved to platform specific code.
set_target_properties(${EXE_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
    COMPILE_FLAGS "/nologo /EHs /FC /Gm- /WX /W4 /wd5208 /wd4505 /wd4127 /wd4047 /wd4024 /DGRAZIE_ASSERT /DGRAZIE_PLT_WIN "
    COMPILE_FLAGS_DEBUG "/Od /MTd /Z7 /Zo /DGRAZIE_DEBUG"
    COMPILE_FLAGS_RELEASE "/O2 /Oi /MT /DGRAZIE_RELEASE"
    LINK_FLAGS "/nologo"
    LINK_FLAGS_DEBUG "/DEBUG:FULL"
)

# TODO(abid): `Bcrypt.lib` must be moved to platform specific code.
target_link_libraries(${EXE_NAME} PUBLIC Bcrypt.lib)

add_custom_target(exec
    COMMAND cd .. && ${CMAKE_CURRENT_BINARY_DIR}/${BIN_DIR}/${EXE_NAME}
)

