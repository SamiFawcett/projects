package main

import (
	"errors"
	"fmt"
	"math"
)

type person struct {
	name string
	age  int
}

func main() {
	fmt.Println("This is the beginning to an amazing project.")

	x := 1
	if x > 6 {
		fmt.Println("greater than 6")
	} else {
		fmt.Println("less than 6")
	}

	var a [5]int
	b := [3]int{1, 2, 3}
	slice := []int{1, 2, 3}
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(slice)
	slice = append(slice, 4)

	verticies := make(map[string]int)
	verticies["triangle"] = 2
	verticies["square"] = 3

	delete(verticies, "square")

	for i := 0; i < 5; i++ {
		fmt.Println(i)
	}

	for index, value := range slice {
		fmt.Println("index", index, "value", value)
	}

	//print map
	for key, value := range verticies {
		fmt.Println(key, value)
		fmt.Println(sum(value, value))
	}

	result, err := sqrt(16)
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println(result)
	}

	p := person{name: "Sami", age: 21}

	fmt.Println(p)

	i := 69
	inc(&i)
	fmt.Println(i)
}

func inc(x *int) {
	*x++
}

func sum(x int, y int) int {
	return x + y
}

//functions can return multiple values
func sqrt(x float64) (float64, error) {
	if x < 0 {
		return 0, errors.New("undefined for negative numbers")
	}
	return math.Sqrt(x), nil
}
