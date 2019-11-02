@if "%DEBUG%" == "" @echo off
@rem ##########################################################################
@rem
@rem  VASSAR_AMOS_V1 startup script for Windows
@rem
@rem ##########################################################################

@rem Set local scope for the variables with windows NT shell
if "%OS%"=="Windows_NT" setlocal

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_BASE_NAME=%~n0
set APP_HOME=%DIRNAME%..

@rem Add default JVM options here. You can also use JAVA_OPTS and VASSAR_AMOS_V1_OPTS to pass JVM options to this script.
set DEFAULT_JVM_OPTS=

@rem Find java.exe
if defined JAVA_HOME goto findJavaFromJavaHome

set JAVA_EXE=java.exe
%JAVA_EXE% -version >NUL 2>&1
if "%ERRORLEVEL%" == "0" goto init

echo.
echo ERROR: JAVA_HOME is not set and no 'java' command could be found in your PATH.
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:findJavaFromJavaHome
set JAVA_HOME=%JAVA_HOME:"=%
set JAVA_EXE=%JAVA_HOME%/bin/java.exe

if exist "%JAVA_EXE%" goto init

echo.
echo ERROR: JAVA_HOME is set to an invalid directory: %JAVA_HOME%
echo.
echo Please set the JAVA_HOME variable in your environment to match the
echo location of your Java installation.

goto fail

:init
@rem Get command-line arguments, handling Windows variants

if not "%OS%" == "Windows_NT" goto win9xME_args

:win9xME_args
@rem Slurp the command line arguments.
set CMD_LINE_ARGS=
set _SKIP=2

:win9xME_args_slurp
if "x%~1" == "x" goto execute

set CMD_LINE_ARGS=%*

:execute
@rem Setup the command line

set CLASSPATH=%APP_HOME%\lib\VASSAR_AMOS_V1-1.0.jar;%APP_HOME%\lib\system-architecture-problems-1.0.jar;%APP_HOME%\lib\vassar-1.0.jar;%APP_HOME%\lib\orekit-1.0.jar;%APP_HOME%\lib\moeaframework-2.12.jar;%APP_HOME%\lib\libthrift-0.13.0-SNAPSHOT.jar;%APP_HOME%\lib\commons-cli-1.2.jar;%APP_HOME%\lib\httpclient-4.5.6.jar;%APP_HOME%\lib\commons-codec-1.10.jar;%APP_HOME%\lib\commons-lang3-3.9.jar;%APP_HOME%\lib\commons-math3-3.6.1.jar;%APP_HOME%\lib\jfreechart-1.0.15.jar;%APP_HOME%\lib\jcommon-1.0.21.jar;%APP_HOME%\lib\junit-4.11.jar;%APP_HOME%\lib\rsyntaxtextarea-2.5.1.jar;%APP_HOME%\lib\combinatoricslib3-3.3.0.jar;%APP_HOME%\lib\combinatoricslib-2.2.jar;%APP_HOME%\lib\jxl-2.6.12.jar;%APP_HOME%\lib\orekit-10.0.jar;%APP_HOME%\lib\jess-7.1p2.jar;%APP_HOME%\lib\pebble-3.1.0.jar;%APP_HOME%\lib\slf4j-simple-1.7.28.jar;%APP_HOME%\lib\slf4j-api-1.7.28.jar;%APP_HOME%\lib\httpcore-4.4.10.jar;%APP_HOME%\lib\javax.annotation-api-1.3.2.jar;%APP_HOME%\lib\xml-apis-1.3.04.jar;%APP_HOME%\lib\itext-2.1.5.jar;%APP_HOME%\lib\hamcrest-core-1.3.jar;%APP_HOME%\lib\log4j-1.2.14.jar;%APP_HOME%\lib\hipparchus-geometry-1.5.jar;%APP_HOME%\lib\hipparchus-ode-1.5.jar;%APP_HOME%\lib\hipparchus-fitting-1.5.jar;%APP_HOME%\lib\hipparchus-optim-1.5.jar;%APP_HOME%\lib\hipparchus-filtering-1.5.jar;%APP_HOME%\lib\hipparchus-stat-1.5.jar;%APP_HOME%\lib\hipparchus-core-1.5.jar;%APP_HOME%\lib\gson-2.8.0.jar;%APP_HOME%\lib\unbescape-1.1.6.RELEASE.jar;%APP_HOME%\lib\commons-logging-1.2.jar;%APP_HOME%\lib\bcmail-jdk14-138.jar;%APP_HOME%\lib\bcprov-jdk14-138.jar

@rem Execute VASSAR_AMOS_V1
"%JAVA_EXE%" %DEFAULT_JVM_OPTS% %JAVA_OPTS% %VASSAR_AMOS_V1_OPTS%  -classpath "%CLASSPATH%" seakers.JavaClient %CMD_LINE_ARGS%

:end
@rem End local scope for the variables with windows NT shell
if "%ERRORLEVEL%"=="0" goto mainEnd

:fail
rem Set variable VASSAR_AMOS_V1_EXIT_CONSOLE if you need the _script_ return code instead of
rem the _cmd.exe /c_ return code!
if  not "" == "%VASSAR_AMOS_V1_EXIT_CONSOLE%" exit 1
exit /b 1

:mainEnd
if "%OS%"=="Windows_NT" endlocal

:omega
