#include "vsip.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>


/**************函数返回值的定义开始**************/
#define TWD_OK                     ( 0)
#define TWD_ERROR		           (-1)
#define TWD_EPARAM                 (-2)
#define TWD_FILE_OPEN_FAILED       (-3)
#define TWD_FILE_READ_ERROE        (-4)
#define TWD_FILE_WRITE_ERROE       (-5)
#define TWD_MEMORY_ALLOCATE_FAILED (-6)
#define TWD_FILE_NO_LOAD           (-7)
/**************函数返回值的定义结束**************/
#ifndef NULL
#define NULL  ((void *)0)
#endif
/**************雷达宏参数的定义开始**************/
#define TAO		7e-6			// 脉冲宽度
#define FS		20e6			// 采样率
#define FL		222e6			// 下限频率
#define BW		6e6				// 带宽
#define PI		VSIP_PI
#define REPEAT	1

#define distance 600            //物体距离
#define lightspeed 3e8          //光速
/**************雷达宏参数的定义结束**************/

/****************************内部接口申明开始****************************/

/*
 * 内部接口：initialize
 * 参数：n_signal_len--线性调频信号的长度
 * 			n_fft_len--FFT的长度
 * 功能：初始化VSIPL
 *			为各个向量分配空间
 *			初始化FFT与IFFT的变换参数
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int initialize(vsip_length n_signal_len, vsip_length n_fft_len);


/*
 * 内部接口：finalize
 * 参数：无
 * 功能：销毁各个向量
 *			销毁FFT与IFFT的参数对象
 *			结束VSIPL
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int finalize(void);

/*
 * 内部接口：vsip_vcreate_hamming_f
 * 参数：length长度
 * 		hint设置，暂未实现
 * 功能：加汉明窗
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
vsip_vview_f* vsip_vcreate_hamming_f(vsip_length length, vsip_memory_hint hint);

/*
 * 内部接口：init_signal
 * 参数：无
 * 功能：初始化回波信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_signal(float t);


/*
 * 内部接口：tw_vaddgauss_f
 * 参数：需要添加高斯白噪声的实向量视图指针
 * 功能：初始化线性调频信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_vaddgauss_f(vsip_vview_f* a, vsip_vview_f* r);


/*
 * 内部接口：init_alt_signal
 * 参数：无
 * 功能：初始化解析信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_alt_signal(void);


/*
 * 内部接口：tw_cvfliplr_f
 * 参数：a--需要被翻褶的向量视图指针
 * 			r--翻褶后的结果向量视图指针
 * 功能：将一个复向量翻褶
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvfliplr_f(vsip_cvview_f* a, vsip_cvview_f* r);


/*
 * 内部接口：init_match_filter
 * 参数：无
 * 功能：初始化匹配滤波器向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_match_filter(void);


/*
 * 内部接口：tw_cvaddwin_f
 * 参数：a--需要被加窗的向量视图指针
 * 			r--加窗后的结果向量视图指针
 * 功能：对复向量加窗函数
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvaddwin_f(vsip_cvview_f* a, vsip_cvview_f* r);


/*
 * 内部接口：tw_cvaddzero_f
 * 参数：a--需要补零的向量视图指针
 * 			r--补零后的结果向量视图指针
 * 功能：对复向量补零
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvaddzero_f(vsip_cvview_f* a, vsip_cvview_f* r);

/*
 * 内部接口：tw_vprint_f
 * 参数：a--待打印的实向量视图指针
 * 功能：串口打印一个实向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_vprint_f(vsip_vview_f* a);


/*
 * 内部接口：tw_cvprint_f
 * 参数：a--待打印的复向量视图指针
 * 功能：串口打印一个复向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvprint_f(vsip_cvview_f* a);

/****************************内部接口申明结束****************************/

/**************************全局变量定义开始**************************/

vsip_vview_f *g_p_vector_time = NULL;										// 时间轴

vsip_vview_f *g_p_vector_signal = NULL;								//回波信号
vsip_vview_f *g_p_vector_gaussnoise = NULL;                         //高斯白噪声
vsip_vview_f *g_p_vector_signal_addgauss = NULL;                    //回波信号、高斯白噪声
vsip_vview_f *g_p_vector_signal_addgauss_hilbert = NULL;            //回波信号、高斯白噪声、希尔伯特变换
vsip_cvview_f *g_p_vector_alt_signal = NULL;                        //解析信号
vsip_cvview_f *g_p_vector_alt_signal_addzero = NULL;						// 解析信号、补零
vsip_cvview_f *g_p_vector_alt_signal_addzero_fft = NULL;					// 解析信号、补零、FFT

vsip_cvview_f *g_p_vector_lfm_signal = NULL;							//线性调频信号

vsip_cvview_f *g_p_vector_match_filter = NULL;								// 匹配滤波器
vsip_cvview_f *g_p_vector_match_filter_addwindows = NULL;					// 匹配滤波器、加窗
vsip_cvview_f *g_p_vector_match_filter_addwindows_addzero = NULL;			// 匹配滤波器、加窗、补零
vsip_cvview_f *g_p_vector_match_filter_addwindows_addzero_fft = NULL;		// 匹配滤波器、加窗、补零、FFT

vsip_cvview_f *g_p_vector_match_filter_out = NULL;							// 脉冲压缩信号
vsip_cvview_f *g_p_vector_match_filter_out_fft = NULL;						// 脉冲压缩信号、FFT


vsip_fft_f *g_p_fft_op_plan = NULL;										// FFT参数
vsip_fft_f *g_p_ifft_op_plan = NULL;										// IFFT参数

struct timeval g_st_timer_begin, g_st_timer_end;							// 计时器
double g_lf_delta_timer = 0.0, g_lf_timer_sum = 0.0;						// 计时器差值与计时累加器



/**************************全局变量定义结束**************************/

int main(int argc,char *argv[])
{
	vsip_scalar_f f_power = 0.0;									// 信号在某一点的功率值
	vsip_scalar_f f_power_dB = 0.0;									// 信号在某一点的功率值（dB）
	vsip_scalar_vi n_index = 0;										// 该功率值对应的索引号
	vsip_length n_signal_len = (vsip_length)(0.5 + TAO * FS);		// 求出信号的点数
	vsip_length n_fft_len = n_signal_len + n_signal_len - 1;		// 循环卷积等于线性卷积，n_fft_len有最小值，并求之
	n_fft_len = (vsip_length)(ceil)( 0.5 + log2(n_fft_len) );		// FFT要求n_fft_len为2的整数次幂
	n_fft_len = (vsip_length)(floor)( 0.5 + pow(2, n_fft_len) );	// 最终求得FFT点数
	printf("\n\n---------------------------------------start---------------------------------------------\n");
	printf("n_signal_len = %ld, n_fft_len = %ld\n", n_signal_len, n_fft_len);
	
	
	// VSIPL的初始化、向量视图对象内存分配、FFT与IFFT对象初始化
	initialize(n_signal_len, n_fft_len);
	printf("initialize has been run\n\n");
	
	//时间偏移量代表相位
	float t=distance/lightspeed;
	// //单步步过的长度,这里为100m
	// float one_dis=100;
	// //存储每步的最大点索引值
	// /*这里结果数组大小并不一定是这个值，30只是代表寻找到30*100=3000m的物体*/
	// int result[30];
	// //i代表步过次数
	// int i=0;


//while(t<3000/lightspeed){
	// 回波信号的初始化
	gettimeofday(&g_st_timer_begin, NULL);
	init_signal(t);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_signal:\n");
	tw_vprint_f(g_p_vector_signal);
	printf("\n");
	
    //回波信号添加高斯白噪声
	gettimeofday(&g_st_timer_begin, NULL);
	tw_vaddgauss_f(g_p_vector_signal,g_p_vector_signal_addgauss);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_signal_addgauss:\n");
	tw_vprint_f(g_p_vector_signal_addgauss);
	printf("\n");

    //希尔伯特滤波器
	gettimeofday(&g_st_timer_begin, NULL);
	init_alt_signal();
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_alt_signal:\n");
	tw_cvprint_f(g_p_vector_alt_signal);
	printf("\n");


	// 匹配滤波器的初始化
	gettimeofday(&g_st_timer_begin, NULL);
	init_match_filter();
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter:\n");
	tw_cvprint_f(g_p_vector_match_filter);
	printf("\n");
	
	
	// 匹配滤波器加窗函数
	gettimeofday(&g_st_timer_begin, NULL);
	tw_cvaddwin_f(g_p_vector_match_filter, g_p_vector_match_filter_addwindows);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter_addwindows:\n");
	tw_cvprint_f(g_p_vector_match_filter_addwindows);
	printf("\n");
	
	
	
	// 解析信号的补零操作
	gettimeofday(&g_st_timer_begin, NULL);
	tw_cvaddzero_f(g_p_vector_alt_signal, g_p_vector_alt_signal_addzero);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_alt_signal_addzero:\n");
	tw_cvprint_f(g_p_vector_alt_signal_addzero);
	printf("\n");
	
	
	// 加窗后的匹配滤波器的补零操作
	gettimeofday(&g_st_timer_begin, NULL);
	tw_cvaddzero_f(g_p_vector_match_filter_addwindows, g_p_vector_match_filter_addwindows_addzero);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter_addwindows_addzero:\n");
	tw_cvprint_f(g_p_vector_match_filter_addwindows_addzero);
	printf("\n");
	
	
	// 解析信号的FFT
	gettimeofday(&g_st_timer_begin, NULL);
	vsip_ccfftop_f(g_p_fft_op_plan, g_p_vector_alt_signal_addzero, g_p_vector_alt_signal_addzero_fft);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_alt_signal_addzero_fft:\n");
	tw_cvprint_f(g_p_vector_alt_signal_addzero_fft);
	printf("\n");
	
	
	// 匹配滤波器的FFT
	gettimeofday(&g_st_timer_begin, NULL);
	vsip_ccfftop_f(g_p_fft_op_plan, g_p_vector_match_filter_addwindows_addzero, g_p_vector_match_filter_addwindows_addzero_fft);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter_addwindows_addzero_fft:\n");
	tw_cvprint_f(g_p_vector_match_filter_addwindows_addzero_fft);
	printf("\n");
	
	
	// 频域相乘
	gettimeofday(&g_st_timer_begin, NULL);
	vsip_cvmul_f(g_p_vector_alt_signal_addzero_fft, g_p_vector_match_filter_addwindows_addzero_fft, g_p_vector_match_filter_out_fft);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter_out_fft:\n");
	tw_cvprint_f(g_p_vector_match_filter_out_fft);
	printf("\n");
	
	// 匹配滤波器输出信号的频域变时域
	gettimeofday(&g_st_timer_begin, NULL);
	vsip_ccfftop_f(g_p_ifft_op_plan, g_p_vector_match_filter_out_fft, g_p_vector_match_filter_out);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("vector_match_filter_out:\n");
	tw_cvprint_f(g_p_vector_match_filter_out);
	printf("\n");

	
	// 求最高点的功率以及索引号
	gettimeofday(&g_st_timer_begin, NULL);
	f_power = vsip_vcmaxmgsqval_f(g_p_vector_match_filter_out, &n_index);
	f_power_dB = 10*log10(f_power);
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("max power = %f = %fdB\n", f_power, f_power_dB);
	printf("p_index = %p = 0x%lx\n", &n_index, (unsigned long int)&n_index);
	printf("max power index = %ld\n", n_index);

	//将结果存储到数组中
	// result[i]=n_index;
	// i++;
	// t+=one_dis/lightspeed;
//}	
	
	
	/*由于这一步迟迟想不出来怎么给一个普遍适用的方法检测结果，但是针对本例子的说明还是可以的
	因此这步里只是用来展示数组在报告中说明思路的，并不是实际的测试结果，由于报告涉及到这部分内容
	所以并未删除，便于后续测试，在报告中有专门分析这个例子得出一个较为模糊的结论，并且把这个作为
	问题并且提出解决方案等在报告中*/

	//把下面打印代码与上面循环相关代码取消注释即可得到报告中的结果

	//打印存储了索引的结果数组（可以反映物体数量和距离，但是暂未想出怎么提取关键信息）
	// for(int i=0;i<30;i++){
	// 	if(i%8==7){
	// 		printf("%d\n",result[i]);
	// 	}
	// 	else{
	// 		printf("%d ",result[i]);
	// 	}
	// }
	// printf("\n");


	// 销毁各个向量视图、销毁FFT与IFFT的参数对象、结束VSIPL
	printf("\n");
	gettimeofday(&g_st_timer_begin, NULL);
	finalize();
	gettimeofday(&g_st_timer_end, NULL);
	g_lf_delta_timer = (double)(g_st_timer_end.tv_sec - g_st_timer_begin.tv_sec) * 1E6
						+ (double)(g_st_timer_end.tv_usec - g_st_timer_begin.tv_usec);
	g_lf_timer_sum = g_lf_timer_sum + g_lf_delta_timer;
	printf("delta_timer : %.6f us\n", g_lf_delta_timer);
	printf("timer_sum : %.6f us\n", g_lf_timer_sum);
	printf("finalize has been run\n\n");
	printf("\n----------------------------------------end---------------------------------------------\n\n");
	return 0;


}


/****************************内部接口定义开始****************************/

/*
 * 内部接口：initialize
 * 参数：n_signal_len--线性调频信号的长度
 * 			n_fft_len--FFT的长度
 * 功能：初始化VSIPL
 *			为各个向量分配空间
 *			初始化FFT与IFFT的变换参数
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int initialize(vsip_length n_signal_len, vsip_length n_fft_len)
{
    int s = vsip_init(NULL);
	if(s!=0)
	{
		printf("initialize: Err, vsip_init failed!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_time = vsip_vcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_time==NULL)
	{
		printf("initialize: Err, g_p_vector_time==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}

    //高斯白噪声
    g_p_vector_gaussnoise = vsip_vcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_gaussnoise==NULL)
	{
		printf("initialize: Err, g_p_vector_gaussnoise==NULL!\n\n");
		return TWD_ERROR;
	}
	
    g_p_vector_signal = vsip_vcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_signal==NULL)
	{
		printf("initialize: Err, g_p_vector_signal==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
    g_p_vector_signal_addgauss = vsip_vcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_signal_addgauss==NULL)
	{
		printf("initialize: Err, g_p_vector_signal_addgauss==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}

    //卷积后长度会增加10，与最后的解析信号无关，这里是为了不丢失数据便于检测设置
    g_p_vector_signal_addgauss_hilbert = vsip_vcreate_f(n_signal_len+10, VSIP_MEM_NONE);
	if(g_p_vector_signal_addgauss_hilbert==NULL)
	{
		printf("initialize: Err,g_p_vector_signal_addgauss_hilbert==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}

    g_p_vector_alt_signal = vsip_cvcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_alt_signal==NULL)
	{
		printf("initialize: Err,g_p_vector_alt_signal==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_lfm_signal = vsip_cvcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_lfm_signal==NULL)
	{
		printf("initialize: Err,g_p_vector_lfm_signal==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}


	g_p_vector_alt_signal_addzero = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_alt_signal_addzero==NULL)
	{
		printf("initialize: Err, g_p_vector_alt_signal_addzero==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_alt_signal_addzero_fft = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_alt_signal_addzero_fft==NULL)
	{
		printf("initialize: Err, g_p_vector_alt_signal_addzero_fft==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter = vsip_cvcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter_addwindows = vsip_cvcreate_f(n_signal_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter_addwindows==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter_addwindows==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter_addwindows_addzero = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter_addwindows_addzero==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter_addwindows_addzero==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter_addwindows_addzero_fft = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter_addwindows_addzero_fft==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter_addwindows_addzero_fft==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter_out = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter_out==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter_out==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	g_p_vector_match_filter_out_fft = vsip_cvcreate_f(n_fft_len, VSIP_MEM_NONE);
	if(g_p_vector_match_filter_out_fft==NULL)
	{
		printf("initialize: Err, g_p_vector_match_filter_out_fft==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
    //初始化FFT正变换参数
    g_p_fft_op_plan = vsip_ccfftop_create_f(n_fft_len, 1.0, VSIP_FFT_FWD, 1, 0);
	if(g_p_fft_op_plan==NULL)
	{
		printf("initialize: Err, g_p_fft_op_plan==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
    //初始化FFT逆变换参数
    g_p_ifft_op_plan= vsip_ccfftop_create_f(n_fft_len, 1.0/n_fft_len, VSIP_FFT_INV, 1, 0);
	if(g_p_ifft_op_plan==NULL)
	{
		printf("initialize: Err, g_p_ifft_op_plan==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}


	return TWD_OK;
}



/*
 * 内部接口：finalize
 * 参数：无
 * 功能：销毁各个向量
 *			销毁FFT与IFFT的参数对象
 *			结束VSIPL
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int finalize(void)
{
    vsip_valldestroy_f(g_p_vector_time);
	g_p_vector_time = NULL;
    vsip_valldestroy_f(g_p_vector_gaussnoise);
    g_p_vector_gaussnoise = NULL;
    vsip_valldestroy_f(g_p_vector_signal);
	g_p_vector_signal = NULL;
    vsip_valldestroy_f(g_p_vector_signal_addgauss);
	g_p_vector_signal_addgauss = NULL;
    vsip_valldestroy_f(g_p_vector_signal_addgauss_hilbert);
	g_p_vector_signal_addgauss_hilbert = NULL;
    vsip_cvalldestroy_f(g_p_vector_alt_signal);
	g_p_vector_alt_signal = NULL;
    

	vsip_cvalldestroy_f(g_p_vector_lfm_signal);
	g_p_vector_lfm_signal = NULL;
	vsip_cvalldestroy_f(g_p_vector_alt_signal_addzero);
	g_p_vector_alt_signal_addzero = NULL;
	vsip_cvalldestroy_f(g_p_vector_alt_signal_addzero_fft);
	g_p_vector_alt_signal_addzero_fft = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter);
	g_p_vector_match_filter = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter_addwindows);
	g_p_vector_match_filter_addwindows = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter_addwindows_addzero);
	g_p_vector_match_filter_addwindows_addzero = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter_addwindows_addzero_fft);
	g_p_vector_match_filter_addwindows_addzero_fft = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter_out);
	g_p_vector_match_filter_out = NULL;
	vsip_cvalldestroy_f(g_p_vector_match_filter_out_fft);
	g_p_vector_match_filter_out_fft = NULL;


	int s = vsip_fft_destroy_f(g_p_fft_op_plan);
	if(s!=0)
	{
		printf("finalize: Err, g_p_fft_op_plan destroy failed!\n\n");
		return TWD_ERROR;
	}
	else
	{
		g_p_fft_op_plan = NULL;
	}
	
    s = vsip_fft_destroy_f(g_p_ifft_op_plan);
	if(s!=0)
	{
		printf("finalize: Err, g_p_ifft_op_plan destroy failed!\n\n");
		return TWD_ERROR;
	}
	else
	{
		g_p_ifft_op_plan = NULL;
	}
    s = vsip_finalize((void *)0);
	if(s!=0)
	{
		printf("finalize: Err, vsip_finalize failed!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	return TWD_OK;
}



/*
 * 内部接口：init_signal
 * 参数：无
 * 功能：初始化回波信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_signal(float t)
{
	vsip_length n = vsip_vgetlength_f(g_p_vector_signal);
	vsip_vview_f *p_vector_temp1 = NULL;
	p_vector_temp1 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp1==NULL)
	{
		printf("init_signal: Err, p_vector_temp1==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	vsip_vview_f *p_vector_temp2 = NULL;
	p_vector_temp2 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp2==NULL)
	{
		printf("init_signal: Err, p_vector_temp2==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	vsip_vview_f *p_vector_temp3 = NULL;
	p_vector_temp3 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp3==NULL)
	{
		printf("init_signal: Err, p_vector_temp3==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}

    vsip_vview_f *p_vector_temp4 = NULL;
	p_vector_temp4 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp4==NULL)
	{
		printf("init_signal: Err, p_vector_temp4==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}

	vsip_vview_f *p_vector_temp5 = NULL;
	p_vector_temp5 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp5==NULL)
	{
		printf("init_signal: Err, p_vector_temp5==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	//t
	vsip_vramp_f(0, 1.0/FS, g_p_vector_time);
	//2*PI*FL*t
	vsip_svmul_f(2*PI*FL, g_p_vector_time, p_vector_temp1);
	//t*t
	vsip_vmul_f(g_p_vector_time, g_p_vector_time, p_vector_temp2);
	//PI*BW*t*t/TAO
	vsip_svmul_f(PI*BW/TAO, p_vector_temp2, p_vector_temp3);
	vsip_vadd_f(p_vector_temp1, p_vector_temp3, p_vector_temp2);

    //不用欧拉变换而是直接取余弦得到回波信号
	vsip_vcos_f(p_vector_temp2,p_vector_temp4);
 
	//同理
	//起始为t代表相位偏移，实际则是通过t不断增加来生成不同相位的线性调频信号，最后得到结果
 	vsip_vramp_f(t, 1.0/FS, g_p_vector_time);
	vsip_svmul_f(2*PI*FL, g_p_vector_time, p_vector_temp1);
	vsip_vmul_f(g_p_vector_time, g_p_vector_time, p_vector_temp2);
	vsip_svmul_f(PI*BW/TAO, p_vector_temp2, p_vector_temp3);
	vsip_vadd_f(p_vector_temp1, p_vector_temp3, p_vector_temp2);
 
	//线性调频信号，用于匹配滤波器
	vsip_veuler_f(p_vector_temp2, g_p_vector_lfm_signal);


	//同理
    //相距600m的物体
    vsip_vramp_f(2*distance/lightspeed, 1.0/FS, g_p_vector_time);
	vsip_svmul_f(2*PI*FL, g_p_vector_time, p_vector_temp1);
	vsip_vmul_f(g_p_vector_time, g_p_vector_time, p_vector_temp2);
	vsip_svmul_f(PI*BW/TAO, p_vector_temp2, p_vector_temp3);
	vsip_vadd_f(p_vector_temp1, p_vector_temp3, p_vector_temp2);
	vsip_vcos_f(p_vector_temp2,p_vector_temp5);
    
    vsip_vadd_f(p_vector_temp4,p_vector_temp5,g_p_vector_signal);//相加得到回波信号

	vsip_valldestroy_f(p_vector_temp1);
	p_vector_temp1 = NULL;
	vsip_valldestroy_f(p_vector_temp2);
	p_vector_temp2 = NULL;
	vsip_valldestroy_f(p_vector_temp3);
	p_vector_temp3 = NULL;
    vsip_valldestroy_f(p_vector_temp4);
	p_vector_temp4 = NULL;
    vsip_valldestroy_f(p_vector_temp5);
	p_vector_temp5 = NULL;

	
	return TWD_OK;
}


/*
 * 内部接口：tw_vaddgauss_f
 * 参数：需要添加高斯白噪声的实向量视图指针
 * 功能：初始化线性调频信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_vaddgauss_f(vsip_vview_f* a, vsip_vview_f* r){
    vsip_length n = vsip_vgetlength_f(g_p_vector_signal);
   	vsip_vview_f *p_vector_temp1 = NULL;
	p_vector_temp1 = vsip_vcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp1==NULL)
	{
		printf("tw_vaddgauss_f: Err, p_vector_temp1==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	//随机数模式创建
    vsip_randstate *state;
    state = vsip_randcreate(0, n, 1, VSIP_PRNG);
    if(state==NULL)
	{
		printf("tw_vaddgauss_f: Err, state==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
    //生成高斯白噪声，振幅待定
    vsip_vrandn_f(state,p_vector_temp1);
    //由于采样的信号为离散信号,采用求和代替积分求能量
    //求平方和
    vsip_scalar_f result1,result2,result3;
    result1=vsip_vsumsqval_f(a);
    result2=vsip_vsumsqval_f(p_vector_temp1);
	//依照能量比值求出振幅
    result3=sqrt(result1/result2);
    vsip_svmul_f(result3,p_vector_temp1,g_p_vector_gaussnoise);
    //打印高斯噪声用于检测
    printf("vector_gaussnoise:\n");
    tw_vprint_f(g_p_vector_gaussnoise);
    vsip_vadd_f(a,g_p_vector_gaussnoise,r);


    //释放内存
    vsip_valldestroy_f(p_vector_temp1);
	p_vector_temp1 = NULL;
    vsip_randdestroy(state);
    state = NULL;
}

/*
 * 内部接口：init_alt_signal
 * 参数：无
 * 功能：初始化解析信号
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_alt_signal(void)
{
    //希尔伯特变换
    /* 创建卷积核 */
    vsip_length N = vsip_vgetlength_f(g_p_vector_signal_addgauss);
    vsip_length M = 11;
    vsip_vview_f *h = vsip_vcreate_f(M, VSIP_MEM_NONE);
    vsip_vputlength_f(h, M);
	//index从0到10，但是实际上是-5到5,因此index为偶数实际对应信号的奇数
	for(int index=0;index<M;index++){
		if(index%2!=0)//如果index为奇数，实际为偶数，则添加零
		{
			vsip_vput_f(h,index,0);
		}
		else//否则为2/(PI*n)
		{
			vsip_vput_f(h,index,2/(PI*(index-5)));
		}
	}



    /* 创建卷积对象 */
    vsip_conv1d_f *conv = vsip_conv1d_create_f(h , VSIP_NONSYM, N, 1, 0, VSIP_SUPPORT_FULL, VSIP_ALG_TIME);

    /* 执行卷积操作 */
    vsip_convolve1d_f(conv, g_p_vector_signal_addgauss, g_p_vector_signal_addgauss_hilbert);

    //打印结果用于检测
    printf("vector_signal_addgauss_hilbert:\n");
    tw_vprint_f(g_p_vector_signal_addgauss_hilbert);

    vsip_length i;
    for(i=0;i<vsip_cvgetlength_f(g_p_vector_alt_signal);i++){
        vsip_cscalar_f x;
        x.r=vsip_vget_f(g_p_vector_signal_addgauss,i);
		//舍弃前五个点以及后五个点
        x.i=vsip_vget_f(g_p_vector_signal_addgauss_hilbert,i+5);
        vsip_cvput_f(g_p_vector_alt_signal,i,x);
    }

    /* 释放内存 */
    vsip_conv1d_destroy_f(conv);
    vsip_valldestroy_f(h);
    return 0;
}




/*
 * 内部接口：tw_cvfliplr_f
 * 参数：a--需要被翻褶的向量视图指针
 * 			r--翻褶后的结果向量视图指针
 * 功能：将一个复向量翻褶
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvfliplr_f(vsip_cvview_f* a, vsip_cvview_f* r)
{
	if(a==NULL)
	{
		printf("tw_cvfliplr_f: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else if(r==NULL)
	{
		printf("tw_cvfliplr_f: Err, r==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_cscalar_f x;
	vsip_index i;
	vsip_length n = vsip_cvgetlength_f(a);
	for(i = 0; i < n; i++)
	{
		x = vsip_cvget_f(a, i);
		vsip_cvput_f(r, n-1-i, x);
	}
	
	return TWD_OK;
}




/*
 * 内部接口：init_match_filter
 * 参数：无
 * 功能：初始化匹配滤波器向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int init_match_filter(void)
{
	vsip_length n = vsip_cvgetlength_f(g_p_vector_lfm_signal);
	vsip_cvview_f *p_vector_temp = NULL;
	p_vector_temp = vsip_cvcreate_f(n, VSIP_MEM_NONE);
	if(p_vector_temp==NULL)
	{
		printf("init_match_filter: Err, p_vector_temp==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	vsip_cvconj_f(g_p_vector_lfm_signal, p_vector_temp);
	tw_cvfliplr_f(p_vector_temp, g_p_vector_match_filter);
	
	vsip_cvalldestroy_f(p_vector_temp);
	p_vector_temp = NULL;
	
	return TWD_OK;
}



/*
 * 内部接口：tw_cvaddwin_f
 * 参数：a--需要被加窗的向量视图指针
 * 			r--加窗后的结果向量视图指针
 * 功能：对复向量加窗函数
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvaddwin_f(vsip_cvview_f* a, vsip_cvview_f* r)
{
	if(a==NULL)
	{
		printf("tw_cvaddwin_f: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else if(r==NULL)
	{
		printf("tw_cvaddwin_f: Err, r==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_length n = vsip_cvgetlength_f(a);
	vsip_vview_f *p_vector_windows = NULL;
	p_vector_windows = vsip_vcreate_hamming_f(n, VSIP_MEM_NONE);
	if(p_vector_windows==NULL)
	{
		printf("tw_cvaddwin_f: Err, p_vector_windows==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
//	printf("vector_windows:\n");
//	tw_vprint_f(p_vector_windows);
//	printf("\n");
	
	vsip_rcvmul_f(p_vector_windows, a, r);
	vsip_valldestroy_f(p_vector_windows);
	p_vector_windows = NULL;
	
	return TWD_OK;
}

vsip_vview_f* vsip_vcreate_hamming_f(vsip_length length, vsip_memory_hint hint) {
    // 创建向量
    vsip_vview_f* hamming = vsip_vcreate_f(length, hint);  
    // 计算Hamming窗的值
    for (vsip_index i = 0; i < length; i++) {
        double angle = 2.0 * PI * i / (length - 1);
        double value = 0.53836 - 0.46164 * cos(angle);
        vsip_vput_f(hamming, i, value);
    }
    return hamming;
}

/*
 * 内部接口：tw_cvaddzero_f
 * 参数：a--需要补零的向量视图指针
 * 			r--补零后的结果向量视图指针
 * 功能：对复向量补零
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvaddzero_f(vsip_cvview_f* a, vsip_cvview_f* r)
{
	if(a==NULL)
	{
		printf("tw_cvaddzero_f: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else if(r==NULL)
	{
		printf("tw_cvaddzero_f: Err, r==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_cscalar_f x;
	vsip_length n = vsip_cvgetlength_f(a);
	vsip_length m = vsip_cvgetlength_f(r);
	vsip_cvview_f *p_vector_zero = NULL;
	p_vector_zero = vsip_cvcreate_f(m-n, VSIP_MEM_NONE);
	if(p_vector_zero==NULL)
	{
		printf("tw_cvaddzero_f: Err, p_vector_zero==NULL!\n\n");
		return TWD_ERROR;
	}
	else
	{
		;
	}
	
	x.r = 0.0;
	x.i = 0.0;
	vsip_cvfill_f(x, p_vector_zero);
	vsip_cvconcatenate_f(a, p_vector_zero, r);
	vsip_cvalldestroy_f(p_vector_zero);
	p_vector_zero = NULL;
	
	return TWD_OK;
}



/*
 * 内部接口：tw_vprint_vi
 * 参数：a--待打印的索引向量视图指针
 * 功能：串口打印一个索引向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_vprint_vi(vsip_vview_vi* a)
{
	if(a==NULL)
	{
		printf("tw_vprint_vi: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_length i, n;
	vsip_scalar_vi x;
	n = vsip_vgetlength_vi(a);
	for(i=0; i<n; i++)
	{
		x = vsip_vget_vi(a,i);
		printf("%ld\t", x);
		if( (i + 1) % 8 == 0 )
		{
			printf("\n");
		}
	}
	printf("\n");
	
	return TWD_OK;
}



/*
 * 内部接口：tw_vprint_f
 * 参数：a--待打印的实向量视图指针
 * 功能：串口打印一个实向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_vprint_f(vsip_vview_f* a)
{
	if(a==NULL)
	{
		printf("tw_vprint_f: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_length i, n;
	vsip_scalar_f x;
	n = vsip_vgetlength_f(a);
	for(i=0; i<n; i++)
	{
		x = vsip_vget_f(a,i);
		printf("%9.4f\t", x);
		if( (i + 1) % 8 == 0 )
		{
			printf("\n");
		}
	}
	printf("\n");
	
	return TWD_OK;
}



/*
 * 内部接口：tw_cvprint_f
 * 参数：a--待打印的复向量视图指针
 * 功能：串口打印一个复向量
 * 返回值：TWD_OK--成功
 *			TWD_ERROR--失败
 *			TWD_EPARAM--参数错误
 */
int tw_cvprint_f(vsip_cvview_f* a)
{
	if(a==NULL)
	{
		printf("tw_cvprint_f: Err, a==NULL!\n\n");
		return TWD_EPARAM;
	}
	else
	{
		;
	}
	
	vsip_length i, n;
	vsip_cscalar_f x;
	n = vsip_cvgetlength_f(a);
	for(i=0; i<n; i++)
	{
		x = vsip_cvget_f(a,i);
		printf("%+9.4f%+9.4fi\t", x.r, x.i);
		if( (i + 1) % 8 == 0 )
		{
			printf("\n");
		}
	}
	printf("\n");
	
	return TWD_OK;
}

/****************************内部接口定义结束****************************/