package dataset

import (
	"fmt"
	"os"
	"bytes"
	"image"
	"image/png"
	"encoding/base64"
)


func PrintImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}


// get the file as an image
func GetImage(filePath string) image.Image {
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	return img
}

// get the pixel data from an image
func DataFromImage(filePath string) (pixels []float64) {
	// read the file
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, err := png.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}

	// create a grayscale image
	bounds := img.Bounds()
	//fmt.Println("Bounds:", bounds)
	gray := image.NewGray(bounds)

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var rgba = img.At(x, y)
			gray.Set(x, y, rgba)
		}
	}
	// make a pixel array
	pixels = make([]float64, len(gray.Pix))
	outputable_pixels := make([]float64, len(gray.Pix))
	// populate the pixel array subtract Pix from 255 because that's how
	// the MNIST database was trained (in reverse)
	for i := 0; i < len(gray.Pix); i++ {
		pixels[i] = (float64(255-gray.Pix[i]) / 255.0 * 0.999) + 0.001
		outputable_pixels[i] = float64(255-gray.Pix[i])
	}
	//fmt.Println("Pixels Array:", outputable_pixels)
	return
}