#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// name of the file which contais the kernel function
#define PROGRAM_FILE "matrix_multiply.cl"
// name of the kernel function
#define KERNEL_FUNC "multiply_matrix"
// number of lines of the first matrix
#define LM1 50
// number of columns of the first matrix and number of
// the second matrix
#define CM 10
// number of columns of the second matrix
#define CM2 70

/**
*  Find a GPU or CPU (device) which is available for the host returning
* the created device
*/
cl_device_id create_device() {
   cl_platform_id platform;
   cl_device_id dev;
   cl_int err;

   // Identify a platform
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Try to access a GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      printf("GPU not found\n");
      // if can't. Try to access a CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   // If there's an error finish the program
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   // return the device id
   return dev;
}

/**
*  Create a program (kernel function) from a file, returning it
* compiled to the caller.
*  Receives a opencl context, a device ID and the name of the file
* which contains the program.
*/
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   // Read program file and place its content into a buffer
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   // gets the program size
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   // sets the end of the buffer
   program_buffer[program_size] = '\0';
   // reads the file content and close it
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   // Create program from file
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   // deallocate the program buffer
   free(program_buffer);

   // Build the read program
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      // Find size of log and print to std output
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      // prints the log with the error informations
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

/**
*  Construct a matrix from a vector.
*  Receives the vector, its length, and the desired column length.
*  The function works well if the relation between the vector length
* and the desired columns length results in an integer number.
*  Returns a pointer to the first element of the matrix
*/
float** vector_to_matrix(float* v, unsigned long int vlen, unsigned long int cv) {
   unsigned long int mlines = vlen/cv;
   unsigned long int i,j,k=0;
   float** r=(float**)malloc(sizeof(float*)*mlines);
   for(i=0;i<mlines;++i)
      r[i]=(float*)malloc(sizeof(float)*cv);

   for(i=0;i<mlines;++i)
      for(j=0;j<cv;++j)
         r[i][j]=v[k++];

   return r;
}

/**
*  Construct a vector from a matrix
*  Receives the matrix, its number of lines, its number of columns and optionally the 
* reference for a variable which will keep the result vector length.
*  Returns a pointer to the initial position of the vector
*/
float* matrix_to_vector(float** m, unsigned long int l, unsigned long int c, unsigned long int *vlen) {
   unsigned long int i,j,k=0;
   float *v = (float*)malloc(sizeof(float)*l*c);
   for(i=0;i<l;++i)
      for(j=0;j<c;++j)
         v[k++]=m[i][j];
   if(vlen != NULL)
      *vlen = l*c;
   return v;
}

/**
*  Auxiliary function used to print a vector.
*  Receives the reference to the vector and its length
*/
void print_vector(float *v, unsigned long int len) {
   unsigned long int i;
   for(i=0;i<len;++i)
      printf("%.2f ",v[i]);
   printf("\n");
}

// MAIN FUNCTION 
int main() {
   // Declaration of OpenCL structures
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel;
   cl_command_queue queue;
   cl_int i, j, err;
   // variables to the number of threads in one block
   // and total numbers of threads, respectively
   unsigned long int local_size, global_size;

   // The data matrices
   float **m1=(float**)malloc(sizeof(float*)*LM1);
   float **m2=(float**)malloc(sizeof(float*)*CM);
   float **res=(float**)malloc(sizeof(float*)*LM1);
   // the vector which will be send to the devices
   cl_mem d_m1, d_m2, d_res;

   // allocates and initializes the data matrices
   for(i=0;i<LM1;++i) {
      m1[i]=(float*)malloc(sizeof(float)*CM);
      for(j=0;j<CM;++j)
         m1[i][j] = 1.0f;
   }

   for(i=0;i<CM;++i) {
      m2[i]=(float*)malloc(sizeof(float)*CM2);
      for(j=0;j<CM2;++j)
         m2[i][j] = 1.0f;
   }

   for(i=0;i<LM1;++i) {
      res[i]=(float*)malloc(sizeof(float)*CM2);
      for(j=0;j<CM2;++j)
         res[i][j] = 0.0f;
   }

   // Create a device and the context
   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   // creates a program and build it
   program = build_program(context, device, PROGRAM_FILE);

   // Create the data buffers size
   unsigned long int m1size = sizeof(float)*LM1*CM;
   unsigned long int m2size = sizeof(float)*CM*CM2;
   unsigned long int res_size = sizeof(float)*LM1*CM2;

   float *vm1, *vm2, *vres;
   // Make a vector with each of the allocated matrices
   vm1 = matrix_to_vector(m1,(int)LM1,(int)CM, NULL);
   vm2 = matrix_to_vector(m2,(int)CM,(int)CM2, NULL);
   vres = matrix_to_vector(res,(int)LM1,(int)CM2,NULL);

   // defines the total number of threads
   global_size = LM1*CM2;
   // defines the number of threads in one block
   local_size = CM2;

   // create the data buffers to be sent to devices
   d_m1 = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, m1size, vm1, &err);
   d_m2 = clCreateBuffer(context, CL_MEM_READ_ONLY |
         CL_MEM_COPY_HOST_PTR, m2size, vm2, &err);
   d_res = clCreateBuffer(context, CL_MEM_READ_WRITE |
         CL_MEM_COPY_HOST_PTR, res_size, vres, &err);
   if(err < 0) {
      perror("Couldn't create a buffer");
      exit(1);   
   };

   // Create a command queue 
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   // Create a kernel
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel");
      exit(1);
   };

   unsigned long int wm1 = CM;
   unsigned long int wm2 = CM2;
   // Sets the kernel arguments
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_m1);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_m2);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_res);
   err |= clSetKernelArg(kernel, 3, sizeof(unsigned long int),(void*) &wm1);
   err |= clSetKernelArg(kernel, 4, sizeof(unsigned long int),(void*) &wm2);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   // Enqueue the created kernel 
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, 
         &local_size, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel PORRA");
      exit(1);
   }

   // Read the kernel's output
   err = clEnqueueReadBuffer(queue, d_res, CL_TRUE, 0, 
         res_size, vres, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

   printf("\n---finished---\n");

   // transforms the result to a matrix and print the values
   printf("====================================\n");
   res = vector_to_matrix(vres,(unsigned int)LM1*CM2,CM2);
   for(i=0;i<LM1;++i) {
      for(j=0;j<CM2;++j)
         printf("%.2f ", res[i][j]);
      printf("\n");
   }
   printf("====================================\n");

   // Deallocating resources
   clReleaseKernel(kernel);
   clReleaseMemObject(d_m1);
   clReleaseMemObject(d_m2);
   clReleaseMemObject(d_res);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   free(m1);
   free(m2);
   free(res);
   free(vm1);
   free(vm2);
   free(vres);
   return 0;
}