QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    common/getOptions.cpp \
    common/logger.cpp \
    common/sampleEngines.cpp \
    common/sampleOptions.cpp \
    sampleuffmnist.cpp
#INCLUDEPATH += /usr/src/tensorrt/samples/common/
INCLUDEPATH += /usr/local/cuda/include/
INCLUDEPATH += /usr/include/aarch64-linux-gnu/

LIBS += -L/usr/lib/aarch64-linux-gnu/
LIBS += -L/usr/local/cuda/lib64/
LIBS += -L/usr/local/cuda-10.0/targets/aarch64-linux/lib/
LIBS += -L/usr/local/cuda-10.0/targets/aarch64-linux/lib/stubs/

LIBS += -lnvinfer
LIBS +=-lnvparsers 
LIBS +=-lnvinfer_plugin 
LIBS +=-lnvonnxparser
LIBS +=-lrt 
LIBS +=-ldl 
LIBS +=-lpthread
LIBS +=-lcudnn 
LIBS +=-lcublas 
LIBS +=-lcudart
LIBS +=-lculibos 

HEADERS += \
    common/argsParser.h \
    common/BatchStream.h \
    common/buffers.h \
    common/common.h \
    common/EntropyCalibrator.h \
    common/ErrorRecorder.h \
    common/getOptions.h \
    common/half.h \
    common/logger.h \
    common/logging.h \
    common/parserOnnxConfig.h \
    common/sampleConfig.h \
    common/sampleEngines.h \
    common/sampleOptions.h \
    common/sampleUtils.h \
    sampleuffmnist.h



