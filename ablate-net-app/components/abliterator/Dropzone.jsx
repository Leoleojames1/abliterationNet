import React from 'react';
import { useDropzone } from 'react-dropzone';
import { cn } from "@/lib/utils";
import { Upload, FileUp, CheckCircle2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';

const Dropzone = React.forwardRef(({ className, onChange, value, accept, disabled, ...props }, ref) => {
  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject
  } = useDropzone({
    accept: accept || {
      'application/parquet': ['.parquet'],
      'text/csv': ['.csv'],
      'application/json': ['.json']
    },
    disabled,
    onDrop: onChange,
    noClick: false,
    ...props
  });

  return (
    <div 
      {...getRootProps()}
      className={cn(
        "relative rounded-lg border-2 border-dashed border-muted-foreground/25 transition-colors",
        "hover:border-muted-foreground/50 hover:bg-muted/50",
        isDragActive && "border-blue-500 bg-blue-500/10",
        isDragAccept && "border-green-500 bg-green-500/10",
        isDragReject && "border-red-500 bg-red-500/10",
        disabled && "cursor-not-allowed opacity-60",
        className
      )}
    >
      <input {...getInputProps()} ref={ref} id="file-input" className="hidden" />
      
      <div className="flex flex-col items-center justify-center p-8 text-center">
        <Upload className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <div className="text-muted-foreground">
          {isDragActive ? (
            <p className="text-lg font-medium">Drop files here...</p>
          ) : (
            <>
              <p className="text-lg font-medium mb-2">
                Drag & drop files here, or click to select
              </p>
              <p className="text-sm">
                Supports .parquet, .csv, and .json files
              </p>
              <Button
                type="button"
                onClick={() => document.getElementById('file-input').click()}
                className="mt-4"
              >
                Select Files
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
});

Dropzone.displayName = "Dropzone";

export const FileList = ({ files }) => {
  if (!files?.length) return null;

  return (
    <div className="mt-4 space-y-2">
      {files.map((file, index) => (
        <div 
          key={index} 
          className="flex items-center gap-2 p-2 rounded bg-muted/50 border border-input"
        >
          <FileUp className="h-4 w-4 text-muted-foreground" />
          <span className="flex-1 text-sm">{file.name}</span>
          {file.status === 'complete' ? (
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          ) : file.status === 'error' ? (
            <AlertCircle className="h-4 w-4 text-red-500" />
          ) : (
            <div className="h-4 w-4 border-2 border-t-primary rounded-full animate-spin" />
          )}
        </div>
      ))}
    </div>
  );
};

export { Dropzone };

