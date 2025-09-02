# MATLAB-CODE-Differentiate-Polynomial-




function Differentiate_polynomial()
    % Let the function be f(x)=x^3-2*x+3 and let x=[0 1 2] then y=[3 2 7]
    % Input 
    x = input('Enter vector of x values: ');
    y = input('Enter vector of y values: ');
    n = length(x);

    % Construct Polynomial via Divided Difference 
    fprintf('\nInterpolating polynomial (Newton form):\n');
    coeff = divided_difference(x, y);  % এখানে Helper Function কে কল করা হয়েছে নিচ থেকে।

    syms t;   % Symbolic Variable t
    P = coeff(1);
    term = 1;
    for i = 2:n
        term = term * (t - x(i-1));
        P = P + coeff(i) * term;
    end
    P = simplify(P);
    pretty(P);  % Symbolic এক্সপ্রেশনটা ASCII Art স্টাইলে সুন্দর দেখানোর জন্য pretty ব্যবহার করা হয়। 




    % প্রশ্নে দেওয়া ফাংশনকে Derivatives করে কমান্ড উইন্ডোতে pretty বা সুন্দরভাবে দেখাবে।
    fprintf('\n--- Derivatives of the Interpolating Polynomial ---\n');
    P1 = simplify(diff(P, t));    % প্রথম ডেরিভেটিভ
    P2 = simplify(diff(P1, t));   % প্রথম ডেরিভেটিভকে দ্বিতীয়বার ডেরিভেটিভ
    P3 = simplify(diff(P2, t));   % দ্বিতীয় ডেরিভেটিভকে আবারও ডেরিভাটিভ

    fprintf('\nFirst derivative P''(x):\n');
    pretty(P1);
    fprintf('\nSecond derivative P''''(x):\n');
    pretty(P2);
    fprintf('\nThird derivative P''''''(x):\n');
    pretty(P3);




    % Plot Polynomial and Derivatives টু-ডাইমেনশন (২*২=ডাবল) গ্রাফে অক্ষ বরাবর নাম দেওয়া ও ডেরিভেটিভের মান বসানো।
    t_vals = linspace(min(x), max(x), 500);  % linspace(a,b,N): a থেকে b পর্যন্ত সমান দূরত্বে Nটা পয়েন্ট।
    y_vals  = double(subs(P,  t, t_vals));   % double(): Symbolic → numeric array এবং subs(P,t,t_vals):তে t এর জায়গায় t_vals বসানো।
    y1_vals = double(subs(P1, t, t_vals));
    y2_vals = double(subs(P2, t, t_vals));
    y3_vals = double(subs(P3, t, t_vals));

    % figure: (x,y)=(২*২) প্লটের ফিগার কেমন হবে তার নির্দেশনাঃ দৈর্ঘ্য-প্রস্থ, কালার কোড ও একটা সাব-গ্রাফের উপর আরেকটা গ্রাফ একে।
    figure;
    subplot(2,2,1);  % ১নং গ্রাফ (ফাংশন নিজেই)
    plot(t_vals, y_vals, 'b', 'LineWidth', 1.5); 
    hold on; 

    plot(x, y, 'ro');  % যেহেতু একই উইন্ডোতে একই পেজে ৪টা প্লটই দেখাতে চাই তাই সাবপ্লট আগে লিখে তারপর প্লট অংশটুকু লিখা হয়েছে। যদি আলাদা আলাদা পেজে বা ৪টা উইন্ডোতে ৪টা প্লট দেখাতে চাও তবে প্লট অংশটুকু আগে লিখবা।
    title('Interpolating Polynomial P(x)');
    grid on;

    subplot(2,2,2);  % ২নং গ্রাফ (১ম ডেরিভেটিভ)
    plot(t_vals, y1_vals, 'm', 'LineWidth', 1.5);
    title(sprintf('First Derivative\nP''(x)'));
    grid on;

    subplot(2,2,3);  % ৩নং গ্রাফ (দ্বিতীয় ডেরিভাটিভ)
    plot(t_vals, y2_vals, 'g', 'LineWidth', 1.5);
    title('Second Derivative P''''(x)');
    grid on;

    subplot(2,2,4);  % ৪নং গ্রাফ (৩য় ভেরিভেটিভ)
    plot(t_vals, y3_vals, 'r', 'LineWidth', 1.5);
    title('Third Derivative P''''''(x)');
    grid on;
end


% কিছু ব্যাসিক জিনিস প্লট বা গ্রাফ সাজানোর জন্যঃ
% 1. hold on: বর্তমান axes-এ পরের plot গুলি নতুন আরেকটি গ্রাফে হবে; না দিলে নতুন plotটি আগের গ্রাফটা মুছে ফেলবে।
% 2. Grid on: ছক কাগজের দাগ গুলো দেখায়; না দিলে একেবারে সাদা পেপারে শুধু গ্রাফটা দেখাবে।
% 3. Linespace(Min, Max, Devider): গ্রাফপেপারের দাগ গুলো সর্বনিম্ন থেকে সর্বোচ্চ সংখ্যা পর্যন্ত মোট কতটি ভাগে বিভক্ত হবে সেটা বোঝায়।
% 4. LineWidth: গ্রাফে ফাংশনের চিত্র কতটুকু মোটা বা সরু হবে সেটার পরিমাণ নির্দেশ করে।
% 5. Colour code: লাল= r, নীল= b, সবুজ= g, মেজেন্টা= m, হলুদ= y, কালো= k, নাম জানিনা= c 
% আর কালারের সাথে যদি o এড করে দেই যেমন ro, bo, go, mo, yo, ko, co তবে কালারের সাথে সার্কেলও তৈরি হবে।


% Helper Function: Divided Difference 
function coeff = divided_difference(x, y)
    n = length(x);
    table = zeros(n,n);
    table(:,1) = y(:);

    for j = 2:n
        for i = 1:(n - j + 1)
            table(i,j) = (table(i+1,j-1) - table(i,j-1)) / (x(i+j-1) - x(i));
        end
    end
    coeff = table(1,:);
end





% একটা আউটপুটসহ উদাহরণঃ ধরি f(x)=x^3-2*x+3 এবং x=0,1,2 তাহলে y=3,2,7 হবে এবং এগুলোই ইনপুট হিসেবে নিবো।
% Differentiate_polynomial
% Enter vector of x values: [0 1 2]
% Enter vector of y values: [3 2 7]

% Interpolating polynomial (Newton form):
%    2
%  3 t  - 4 t + 3

% --- Derivatives of the Interpolating Polynomial ---

% First derivative P'(x):
%  6 t - 4

%  derivative P''(x):
%  6

% Third derivative P'''(x):
% 0

