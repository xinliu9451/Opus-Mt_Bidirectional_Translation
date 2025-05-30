#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "CTranslate2::ctranslate2" for configuration "Release"
set_property(TARGET CTranslate2::ctranslate2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(CTranslate2::ctranslate2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libctranslate2.so.4.5.0"
  IMPORTED_SONAME_RELEASE "libctranslate2.so.4"
  )

list(APPEND _IMPORT_CHECK_TARGETS CTranslate2::ctranslate2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_CTranslate2::ctranslate2 "${_IMPORT_PREFIX}/lib/libctranslate2.so.4.5.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
