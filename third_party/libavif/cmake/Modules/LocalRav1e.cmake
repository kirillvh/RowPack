set(AVIF_RAV1E_GIT_TAG v0.8.1)
set(AVIF_CORROSION_GIT_TAG v0.6.1)
set(AVIF_CARGOC_GIT_TAG v0.10.20)
set(AVIF_RAV1E_ENABLE_ASM
    "AUTO"
    CACHE STRING "Enable rav1e assembly optimizations. AUTO enables them only when NASM or YASM is available."
)
set_property(CACHE AVIF_RAV1E_ENABLE_ASM PROPERTY STRINGS AUTO ON OFF)

set(RAV1E_LIB_FILENAME
    "${AVIF_SOURCE_DIR}/ext/rav1e/build.libavif/usr/lib/${AVIF_LIBRARY_PREFIX}rav1e${CMAKE_STATIC_LIBRARY_SUFFIX}"
)

if(EXISTS "${RAV1E_LIB_FILENAME}")
    message(STATUS "libavif(AVIF_CODEC_RAV1E=LOCAL): compiled rav1e library found at ${RAV1E_LIB_FILENAME}")
    add_library(rav1e::rav1e STATIC IMPORTED)
    set_target_properties(rav1e::rav1e PROPERTIES IMPORTED_LOCATION "${RAV1E_LIB_FILENAME}" IMPORTED_SONAME rav1e AVIF_LOCAL ON)
    target_include_directories(rav1e::rav1e INTERFACE "${AVIF_SOURCE_DIR}/ext/rav1e/build.libavif/usr/include/rav1e")
else()
    message(
        STATUS "libavif(AVIF_CODEC_RAV1E=LOCAL): compiled rav1e library not found at ${RAV1E_LIB_FILENAME}; using FetchContent"
    )
    if(EXISTS "${AVIF_SOURCE_DIR}/ext/rav1e")
        message(STATUS "libavif(AVIF_CODEC_RAV1E=LOCAL): ext/rav1e found; using as FetchContent SOURCE_DIR")
        set(FETCHCONTENT_SOURCE_DIR_RAV1E "${AVIF_SOURCE_DIR}/ext/rav1e")
        message(CHECK_START "libavif(AVIF_CODEC_RAV1E=LOCAL): configuring rav1e")
    else()
        message(CHECK_START "libavif(AVIF_CODEC_RAV1E=LOCAL): fetching and configuring rav1e")
    endif()

    FetchContent_Declare(
        Corrosion
        EXCLUDE_FROM_ALL
        GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
        GIT_TAG ${AVIF_CORROSION_GIT_TAG}
        GIT_SHALLOW ON
    )

    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(Rust_CARGO_TARGET "aarch64-apple-darwin")
        endif()
    endif()

    FetchContent_MakeAvailable(Corrosion)

    find_program(
        CARGO_CINSTALL cargo-cinstall
        HINTS "$ENV{HOME}/.cargo/bin"
              "$ENV{USERPROFILE}/.cargo/bin"
              ${CMAKE_CURRENT_BINARY_DIR}
              ${CMAKE_CURRENT_BINARY_DIR}/Release
              ${CMAKE_BINARY_DIR}
              ${CMAKE_BINARY_DIR}/Release
    )
    find_program(
        CARGO_CBUILD cargo-cbuild
        HINTS "$ENV{HOME}/.cargo/bin"
              "$ENV{USERPROFILE}/.cargo/bin"
              ${CMAKE_CURRENT_BINARY_DIR}
              ${CMAKE_CURRENT_BINARY_DIR}/Release
              ${CMAKE_BINARY_DIR}
              ${CMAKE_BINARY_DIR}/Release
    )

    if(WIN32 AND CARGO_CINSTALL)
        string(FIND "${CARGO_CINSTALL}" "$env:USERPROFILE" USERPROFILE_POWERSHELL_POS)
        if(USERPROFILE_POWERSHELL_POS EQUAL 0 AND DEFINED ENV{USERPROFILE})
            string(REPLACE "$env:USERPROFILE" "$ENV{USERPROFILE}" CARGO_CINSTALL "${CARGO_CINSTALL}")
            file(TO_CMAKE_PATH "${CARGO_CINSTALL}" CARGO_CINSTALL)
            set(CARGO_CINSTALL "${CARGO_CINSTALL}" CACHE FILEPATH "Path to cargo-cinstall" FORCE)
        endif()

        string(FIND "${CARGO_CINSTALL}" "%USERPROFILE%" USERPROFILE_CMD_POS)
        if(USERPROFILE_CMD_POS EQUAL 0 AND DEFINED ENV{USERPROFILE})
            string(REPLACE "%USERPROFILE%" "$ENV{USERPROFILE}" CARGO_CINSTALL "${CARGO_CINSTALL}")
            file(TO_CMAKE_PATH "${CARGO_CINSTALL}" CARGO_CINSTALL)
            set(CARGO_CINSTALL "${CARGO_CINSTALL}" CACHE FILEPATH "Path to cargo-cinstall" FORCE)
        endif()
    endif()

    if(WIN32 AND CARGO_CBUILD)
        string(FIND "${CARGO_CBUILD}" "$env:USERPROFILE" USERPROFILE_POWERSHELL_POS)
        if(USERPROFILE_POWERSHELL_POS EQUAL 0 AND DEFINED ENV{USERPROFILE})
            string(REPLACE "$env:USERPROFILE" "$ENV{USERPROFILE}" CARGO_CBUILD "${CARGO_CBUILD}")
            file(TO_CMAKE_PATH "${CARGO_CBUILD}" CARGO_CBUILD)
            set(CARGO_CBUILD "${CARGO_CBUILD}" CACHE FILEPATH "Path to cargo-cbuild" FORCE)
        endif()

        string(FIND "${CARGO_CBUILD}" "%USERPROFILE%" USERPROFILE_CMD_POS)
        if(USERPROFILE_CMD_POS EQUAL 0 AND DEFINED ENV{USERPROFILE})
            string(REPLACE "%USERPROFILE%" "$ENV{USERPROFILE}" CARGO_CBUILD "${CARGO_CBUILD}")
            file(TO_CMAKE_PATH "${CARGO_CBUILD}" CARGO_CBUILD)
            set(CARGO_CBUILD "${CARGO_CBUILD}" CACHE FILEPATH "Path to cargo-cbuild" FORCE)
        endif()
    endif()

    if(CARGO_CINSTALL)
        add_executable(cargo-cinstall IMPORTED GLOBAL)
        set_property(TARGET cargo-cinstall PROPERTY IMPORTED_LOCATION ${CARGO_CINSTALL})
    endif()

    if(NOT CARGO_CBUILD AND NOT TARGET cargo-cinstall)
        FetchContent_Declare(
            cargoc
            EXCLUDE_FROM_ALL
            GIT_REPOSITORY https://github.com/lu-zero/cargo-c.git
            GIT_TAG "${AVIF_CARGOC_GIT_TAG}"
            GIT_SHALLOW ON
        )
        FetchContent_MakeAvailable(cargoc)

        corrosion_import_crate(
            MANIFEST_PATH ${cargoc_SOURCE_DIR}/Cargo.toml PROFILE release IMPORTED_CRATES MYVAR_IMPORTED_CRATES FEATURES
            vendored-openssl
        )

        set(CARGO_CINSTALL $<TARGET_FILE:cargo-cinstall>)
    endif()

    if(CARGO_CBUILD)
        set(RAV1E_CARGO_C_COMMAND ${CARGO_CBUILD})
        set(RAV1E_CARGO_C_SUBCOMMAND cbuild)
        set(RAV1E_CARGO_C_DEPENDS)
    else()
        set(RAV1E_CARGO_C_COMMAND ${CARGO_CINSTALL})
        set(RAV1E_CARGO_C_SUBCOMMAND cinstall)
        set(RAV1E_CARGO_C_DEPENDS cargo-cinstall)
    endif()

    FetchContent_Declare(
        rav1e
        EXCLUDE_FROM_ALL
        GIT_REPOSITORY https://github.com/xiph/rav1e.git
        GIT_TAG "${AVIF_RAV1E_GIT_TAG}"
        GIT_SHALLOW ON
    )
    FetchContent_MakeAvailable(rav1e)

    set(RAV1E_LIB_FILENAME
        ${CMAKE_CURRENT_BINARY_DIR}/ext/rav1e/usr/lib/${CMAKE_STATIC_LIBRARY_PREFIX}rav1e${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
    set(RAV1E_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/ext/rav1e/usr/include/rav1e")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ext/rav1e/usr/lib" "${RAV1E_INCLUDE_DIR}")
    set(RAV1E_CBUILD_COPY_COMMANDS)
    if(CARGO_CBUILD)
        set(RAV1E_CBUILD_OUTPUT_DIR "${rav1e_SOURCE_DIR}/target/${Rust_CARGO_TARGET_CACHED}/release")
        set(RAV1E_CBUILD_LIB_FILENAME "${RAV1E_CBUILD_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}rav1e${CMAKE_STATIC_LIBRARY_SUFFIX}")
        set(RAV1E_CBUILD_HEADER_FILENAME "${RAV1E_CBUILD_OUTPUT_DIR}/include/rav1e/rav1e.h")
        list(
            APPEND
            RAV1E_CBUILD_COPY_COMMANDS
            COMMAND
            ${CMAKE_COMMAND}
            -E
            make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/ext/rav1e/usr/lib"
            "${RAV1E_INCLUDE_DIR}"
            COMMAND
            ${CMAKE_COMMAND}
            -E
            copy_if_different
            "${RAV1E_CBUILD_LIB_FILENAME}"
            "${RAV1E_LIB_FILENAME}"
            COMMAND
            ${CMAKE_COMMAND}
            -E
            copy_if_different
            "${RAV1E_CBUILD_HEADER_FILENAME}"
            "${RAV1E_INCLUDE_DIR}/rav1e.h"
        )
    endif()

    set(RAV1E_CARGO_FEATURE_ARGS)
    set(RAV1E_ENABLE_ASM_VALUE true)
    if(AVIF_RAV1E_ENABLE_ASM STREQUAL "ON")
        set(RAV1E_ENABLE_ASM_VALUE true)
    elseif(AVIF_RAV1E_ENABLE_ASM STREQUAL "OFF")
        set(RAV1E_ENABLE_ASM_VALUE false)
    else()
        find_program(RAV1E_NASM_EXECUTABLE NAMES nasm)
        find_program(RAV1E_YASM_EXECUTABLE NAMES yasm)
        if(RAV1E_NASM_EXECUTABLE OR RAV1E_YASM_EXECUTABLE)
            set(RAV1E_ENABLE_ASM_VALUE true)
        else()
            set(RAV1E_ENABLE_ASM_VALUE false)
            message(STATUS "libavif(AVIF_CODEC_RAV1E=LOCAL): NASM/YASM not found; configuring rav1e without asm")
        endif()
    endif()
    if(NOT RAV1E_ENABLE_ASM_VALUE)
        list(APPEND RAV1E_CARGO_FEATURE_ARGS --no-default-features --features=capi,threading,signal_support,git_version)
    endif()

    set(RAV1E_ENVVARS)
    if(AVIF_OPTIMIZE_RAV1E_FOR_SIZE)
        set(RAV1E_ENVVARS "CARGO_PROFILE_RELEASE_DEBUG=0" "CARGO_PROFILE_RELEASE_STRIP=true"
                          "CARGO_PROFILE_PROFILE_RELEASE_OPT_LEVEL=\"s\"" "CARGO_PROFILE_RELEASE_INCREMENTAL=false"
        )
    endif()
    if(CMAKE_C_IMPLICIT_LINK_DIRECTORIES MATCHES "alpine-linux-musl")
        list(APPEND RAV1E_ENVVARS "RUSTFLAGS=-C link-args=-Wl,-z,stack-size=2097152 -C target-feature=-crt-static")
    endif()
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_SYSROOT)
        list(APPEND RAV1E_ENVVARS "SDKROOT=${CMAKE_OSX_SYSROOT}")
    endif()
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_DEPLOYMENT_TARGET)
        list(APPEND RAV1E_ENVVARS "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif()
    if(Rust_COMPILER_CACHED)
        set(RAV1E_RUSTC "${Rust_COMPILER_CACHED}")
    elseif(Rust_COMPILER)
        set(RAV1E_RUSTC "${Rust_COMPILER}")
    elseif(_CORROSION_RUSTC)
        set(RAV1E_RUSTC "${_CORROSION_RUSTC}")
    endif()
    if(Rust_CARGO_CACHED)
        set(RAV1E_CARGO "${Rust_CARGO_CACHED}")
    elseif(Rust_CARGO)
        set(RAV1E_CARGO "${Rust_CARGO}")
    elseif(_CORROSION_CARGO)
        set(RAV1E_CARGO "${_CORROSION_CARGO}")
    endif()
    if(RAV1E_RUSTC)
        file(TO_CMAKE_PATH "${RAV1E_RUSTC}" RAV1E_RUSTC)
        get_filename_component(RAV1E_RUST_BIN_DIR "${RAV1E_RUSTC}" DIRECTORY)
        set(RAV1E_PATH "$ENV{PATH}")
        if(WIN32)
            string(REPLACE ";" "\$<SEMICOLON>" RAV1E_PATH "${RAV1E_PATH}")
            set(RAV1E_PATH "${RAV1E_RUST_BIN_DIR}$<SEMICOLON>${RAV1E_PATH}")
        else()
            set(RAV1E_PATH "${RAV1E_RUST_BIN_DIR}:${RAV1E_PATH}")
        endif()
        list(APPEND RAV1E_ENVVARS "PATH=${RAV1E_PATH}" "RUSTC=${RAV1E_RUSTC}")
    endif()
    if(RAV1E_CARGO)
        file(TO_CMAKE_PATH "${RAV1E_CARGO}" RAV1E_CARGO)
        list(APPEND RAV1E_ENVVARS "CARGO=${RAV1E_CARGO}")
    endif()

    add_custom_target(
        rav1e
        COMMAND ${CMAKE_COMMAND} -E env ${RAV1E_ENVVARS} -- ${RAV1E_CARGO_C_COMMAND} ${RAV1E_CARGO_C_SUBCOMMAND}
                ${RAV1E_CARGO_FEATURE_ARGS} -v --release --library-type=staticlib --prefix=/usr --target ${Rust_CARGO_TARGET_CACHED}
                --destdir ${CMAKE_CURRENT_BINARY_DIR}/ext/rav1e
        ${RAV1E_CBUILD_COPY_COMMANDS}
        DEPENDS ${RAV1E_CARGO_C_DEPENDS}
        BYPRODUCTS ${RAV1E_LIB_FILENAME}
        USES_TERMINAL
        WORKING_DIRECTORY ${rav1e_SOURCE_DIR}
    )
    set(RAV1E_FOUND ON)

    add_library(rav1e::rav1e STATIC IMPORTED)
    add_dependencies(rav1e::rav1e rav1e)
    target_link_libraries(rav1e::rav1e INTERFACE "${Rust_CARGO_TARGET_LINK_NATIVE_LIBS}")
    target_link_options(rav1e::rav1e INTERFACE "${Rust_CARGO_TARGET_LINK_OPTIONS}")
    set_target_properties(rav1e::rav1e PROPERTIES IMPORTED_LOCATION "${RAV1E_LIB_FILENAME}" AVIF_LOCAL ON FOLDER "ext/rav1e")
    target_include_directories(rav1e::rav1e INTERFACE "${RAV1E_INCLUDE_DIR}")

    message(CHECK_PASS "complete")
endif()
