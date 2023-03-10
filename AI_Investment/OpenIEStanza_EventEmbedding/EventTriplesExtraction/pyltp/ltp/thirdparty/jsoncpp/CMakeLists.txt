# vim: et ts=4 sts=4 sw=4 tw=0

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)
PROJECT(jsoncpp)

OPTION(BUILD_SHARED_LIBS "Build jsoncpp_lib as a shared library." OFF)
OPTION(BUILD_STATIC_LIBS "Build jsoncpp_lib static library." ON)

# Ensures that CMAKE_BUILD_TYPE is visible in cmake-gui on Unix
IF(NOT WIN32)
    IF(NOT CMAKE_BUILD_TYPE)
        SET(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage."
            FORCE)
    ENDIF()
ENDIF()

# Enable runtime search path support for dynamic libraries on OSX
IF(APPLE)
    SET(CMAKE_MACOSX_RPATH 1)
ENDIF()

SET(DEBUG_LIBNAME_SUFFIX "" CACHE STRING "Optional suffix to append to the library name for a debug build")
SET(LIB_SUFFIX "" CACHE STRING "Optional arch-dependent suffix for the library installation directory")

SET(RUNTIME_INSTALL_DIR bin
    CACHE PATH "Install dir for executables and dlls")
SET(ARCHIVE_INSTALL_DIR lib${LIB_SUFFIX}
    CACHE PATH "Install dir for static libraries")
SET(LIBRARY_INSTALL_DIR lib${LIB_SUFFIX}
    CACHE PATH "Install dir for shared libraries")
SET(INCLUDE_INSTALL_DIR include
    CACHE PATH "Install dir for headers")
SET(PACKAGE_INSTALL_DIR lib${LIB_SUFFIX}/cmake
    CACHE PATH "Install dir for cmake package config files")
MARK_AS_ADVANCED( RUNTIME_INSTALL_DIR ARCHIVE_INSTALL_DIR INCLUDE_INSTALL_DIR PACKAGE_INSTALL_DIR )

# Set variable named ${VAR_NAME} to value ${VALUE}
FUNCTION(set_using_dynamic_name VAR_NAME VALUE)
    SET( "${VAR_NAME}" "${VALUE}" PARENT_SCOPE)
ENDFUNCTION()

# Extract major, minor, patch from version text
# Parse a version string "X.Y.Z" and outputs
# version parts in ${OUPUT_PREFIX}_MAJOR, _MINOR, _PATCH.
# If parse succeeds then ${OUPUT_PREFIX}_FOUND is TRUE.
MACRO(jsoncpp_parse_version VERSION_TEXT OUPUT_PREFIX)
    SET(VERSION_REGEX "[0-9]+\\.[0-9]+\\.[0-9]+(-[a-zA-Z0-9_]+)?")
    IF( ${VERSION_TEXT} MATCHES ${VERSION_REGEX} )
        STRING(REGEX MATCHALL "[0-9]+|-([A-Za-z0-9_]+)" VERSION_PARTS ${VERSION_TEXT})
        LIST(GET VERSION_PARTS 0 ${OUPUT_PREFIX}_MAJOR)
        LIST(GET VERSION_PARTS 1 ${OUPUT_PREFIX}_MINOR)
        LIST(GET VERSION_PARTS 2 ${OUPUT_PREFIX}_PATCH)
        set_using_dynamic_name( "${OUPUT_PREFIX}_FOUND" TRUE )
    ELSE( ${VERSION_TEXT} MATCHES ${VERSION_REGEX} )
        set_using_dynamic_name( "${OUPUT_PREFIX}_FOUND" FALSE )
    ENDIF()
ENDMACRO()

# Read out version from "version" file
#FILE(STRINGS "version" JSONCPP_VERSION)
#SET( JSONCPP_VERSION_MAJOR X )
#SET( JSONCPP_VERSION_MINOR Y )
#SET( JSONCPP_VERSION_PATCH Z )
SET( JSONCPP_VERSION 1.7.7 )
jsoncpp_parse_version( ${JSONCPP_VERSION} JSONCPP_VERSION )
#IF(NOT JSONCPP_VERSION_FOUND)
#    MESSAGE(FATAL_ERROR "Failed to parse version string properly. Expect X.Y.Z")
#ENDIF(NOT JSONCPP_VERSION_FOUND)
SET( JSONCPP_SOVERSION 11 )
SET( JSONCPP_USE_SECURE_MEMORY "0" CACHE STRING "-D...=1 to use memory-wiping allocator for STL" )

#MESSAGE(STATUS "JsonCpp Version: ${JSONCPP_VERSION_MAJOR}.${JSONCPP_VERSION_MINOR}.${JSONCPP_VERSION_PATCH}")
# File version.h is only regenerated on CMake configure step
CONFIGURE_FILE( "${PROJECT_SOURCE_DIR}/src/lib_json/version.h.in"
                "${PROJECT_SOURCE_DIR}/include/json/version.h"
                NEWLINE_STYLE UNIX )
CONFIGURE_FILE( "${PROJECT_SOURCE_DIR}/version.in"
                "${PROJECT_SOURCE_DIR}/version"
                NEWLINE_STYLE UNIX )

macro(UseCompilationWarningAsError)
	if ( MSVC )
        # Only enabled in debug because some old versions of VS STL generate
        # warnings when compiled in release configuration.
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /WX ")
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    if (JSONCPP_WITH_STRICT_ISO)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic-errors")
    endif ()
	endif()
endmacro()

# Include our configuration header
INCLUDE_DIRECTORIES( ${jsoncpp_SOURCE_DIR}/include )

if ( MSVC )
    # Only enabled in debug because some old versions of VS STL generate
    # unreachable code warning when compiled in release configuration.
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W4 ")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # using regular Clang or AppleClang
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wconversion -Wshadow -Werror=conversion -Werror=sign-compare")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wconversion -Wshadow -Wextra")
  # not yet ready for -Wsign-conversion

  if (JSONCPP_WITH_STRICT_ISO AND NOT JSONCPP_WITH_WARNING_AS_ERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=conversion -pedantic")
  endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  #??using Intel compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wconversion -Wshadow -Wextra -Werror=conversion")

  if (JSONCPP_WITH_STRICT_ISO AND NOT JSONCPP_WITH_WARNING_AS_ERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")
  endif ()
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

IF(JSONCPP_WITH_WARNING_AS_ERROR)
    UseCompilationWarningAsError()
ENDIF()

IF(JSONCPP_WITH_PKGCONFIG_SUPPORT)
	CONFIGURE_FILE(
		"pkg-config/jsoncpp.pc.in"
		"pkg-config/jsoncpp.pc"
		@ONLY)
	INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/pkg-config/jsoncpp.pc"
		DESTINATION "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/pkgconfig")
ENDIF()

IF(JSONCPP_WITH_CMAKE_PACKAGE)
        INSTALL(EXPORT jsoncpp
                DESTINATION ${PACKAGE_INSTALL_DIR}/jsoncpp
                FILE        jsoncppConfig.cmake)
ENDIF()

# Build the different applications
ADD_SUBDIRECTORY( src )

#install the includes
ADD_SUBDIRECTORY( include )
