cmake_minimum_required(VERSION 2.4)
if(COMMAND CMAKE_POLICY)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND CMAKE_POLICY)

PROJECT(BayesianImageRegistration)

FIND_PACKAGE(ITK)
IF(ITK_FOUND)
INCLUDE(${USE_ITK_FILE})
ENDIF(ITK_FOUND)

SET(EIGEN_DIR "" CACHE PATH "")
SET(BOOST_DIR "" CACHE PATH "")

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${EIGEN_DIR}
  ${BOOST_DIR}
)

SET(BayesianImageRegistrationHeaders
)

SET(BayesianImageRegistrationSrcs
  ${BayesianImageRegistration_SOURCE_DIR}/src/Main.cxx
)

ADD_EXECUTABLE( BayesianImageRegistration ${BayesianImageRegistrationHeaders} ${BayesianImageRegistrationSrcs})

TARGET_LINK_LIBRARIES( BayesianImageRegistration
  ${ITK_LIBRARIES}
  ${BOOST_DIR}/stage/lib/libboost_program_options.a
)
