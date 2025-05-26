# Batch File Design Document for Technical Agents  
**Version 1.0**  
*Best Practices for Robust, Maintainable, and Error-Resistant Batch Scripts*  

---

## 1. Core Principles  
### 1.1 Purpose of Batch Files  
- Automate repetitive tasks (e.g., file management, deployments, system checks).  
- Ensure consistency across environments.  
- Simplify complex workflows.  

### 1.2 Fundamental Concepts  
- **Delayed Expansion**: Use `!var!` for variables modified within code blocks (e.g., loops, conditionals).  
- **Error Handling**: Check `%ERRORLEVEL%` and validate critical operations.  
- **Path Safety**: Always quote paths to handle spaces and special characters.  

---

## 2. Best Practices  
### 2.1 Code Structure  
| **Do** | **Don’t** |  
|--------|-----------|  
| Use subroutines (`CALL :label`) for modular code | Nest `IF`/`FOR` blocks excessively |  
| Separate configuration, validation, and execution logic | Write monolithic scripts without sections |  
| Use `SETLOCAL`/`ENDLOCAL` to isolate variables | Allow variable leakage between scripts |  

**Example Structure**:  
```batch
@echo OFF
SETLOCAL ENABLEDELAYEDEXPANSION

REM === Configuration ===
SET "PROJECT_ROOT=%~dp0.."

REM === Validation ===
CALL :ValidateInputs

REM === Main Logic ===
CALL :ProcessFiles

REM === Cleanup ===
ENDLOCAL
EXIT /B %ERRORLEVEL%

:ValidateInputs
  IF NOT EXIST "!PROJECT_ROOT!" (
    ECHO ERROR: Project root missing
    EXIT /B 1
  )
EXIT /B 0
```

### 2.2 Variable Management  
- **Declare Variables Clearly**: Use uppercase and descriptive names.  
  ```batch
  SET "LOG_DIR=%PROJECT_ROOT%\logs"
  ```  
- **Avoid Global Variables**: Use `SETLOCAL` to scope variables.  
- **Dynamic Variables**: Use `CALL SET` for computed values.  
  ```batch
  FOR %%F IN (*.txt) DO (CALL SET "FILENAME=%%~nF")
  ```  

### 2.3 Error Handling  
- **Check Critical Operations**:  
  ```batch
  MKDIR "!OUTPUT_DIR!"  
  IF %ERRORLEVEL% NEQ 0 (  
    ECHO Failed to create directory  
    EXIT /B 1  
  )  
  ```  
- **Use Explicit Exit Codes**:  
  ```batch
  EXIT /B 0  REM Success
  EXIT /B 1  REM General error
  ```  

---

## 3. Common Pitfalls & Solutions  
| **Issue** | **Solution** |  
|-----------|--------------|  
| Variables not updating in loops | Use delayed expansion (`!var!`) |  
| Script fails on paths with spaces | Always wrap paths in quotes |  
| Nested `IF`/`ELSE` unreadable | Use subroutines (`CALL :label`) |  
| Accidental variable overwrites | Use `SETLOCAL ENABLEDELAYEDEXPANSION` |  

**Example Fix**:  
```batch
REM Bad: Mixing %var% and !var!
IF "%VAR%"=="true" SET "FLAG=!NEW_VALUE!"

REM Good: Consistent delayed expansion
IF "!VAR!"=="true" SET "FLAG=!NEW_VALUE!"
```

---

## 4. Advanced Techniques  
### 4.1 Temporary Files  
- Create and clean up temp files safely:  
  ```batch
  SET "TEMP_FILE=%TEMP%\temp_%RANDOM%.txt"  
  (ECHO Temporary content) > "!TEMP_FILE!"  
  DEL "!TEMP_FILE!" 2>nul  
  ```  

### 4.2 Logging  
- Redirect output to log files:  
  ```batch
  CALL :MainLogic > "!LOG_DIR!\run_%DATE%.log" 2>&1  
  ```  

### 4.3 User Input  
- Sanitize inputs to prevent injection:  
  ```batch
  SET /P "USER_INPUT=Enter value: "  
  IF "!USER_INPUT!"=="" EXIT /B 1  
  ```  

---

## 5. Testing & Debugging  
### 5.1 Debugging Checklist  
1. Run with `@echo ON` to trace execution.  
2. Add `ECHO` statements to log variable states.  
3. Test edge cases (e.g., missing files, empty inputs).  

### 5.2 Tools  
- **Notepad++**: Syntax highlighting for batch files.  
- **Batch Linter**: Validate script structure (e.g., [Batch VS Code extensions](https://marketplace.visualstudio.com/items?itemName=shakram02.bash-beautify)).  

---

## 6. Example Scripts  
### 6.1 Simple File Backup  
```batch
@echo OFF
SETLOCAL

SET "SOURCE_DIR=C:\data"
SET "BACKUP_DIR=C:\backups\data_%DATE%"

IF NOT EXIST "!SOURCE_DIR!" (
  ECHO Source directory missing
  EXIT /B 1
)

ROBOCOPY "!SOURCE_DIR!" "!BACKUP_DIR!" /MIR
IF %ERRORLEVEL% GTR 3 (
  ECHO Backup failed
  EXIT /B 1
)

ECHO Backup successful
ENDLOCAL
```  

### 6.2 Troubleshooting Example  
**Problem**: Script fails when `PROJECT_ROOT` has spaces.  
**Fix**:  
```batch
REM Before (fails):
SET PROJECT_ROOT=C:\My Project

REM After (works):
SET "PROJECT_ROOT=C:\My Project"
```  

---

## 7. Checklist for Agents  
- [ ] Use `SETLOCAL ENABLEDELAYEDEXPANSION`  
- [ ] Quote all paths (`"!PATH!"`)  
- [ ] Validate inputs and critical operations  
- [ ] Use subroutines instead of nested conditionals  
- [ ] Test with spaces in paths and empty inputs  

--- 
 



*Documentation is a first-class citizen in maintainable code.* 🛠️